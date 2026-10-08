"""
Custom model loaders for medical VLMs that require special loading logic.
Each wrapper follows the CLIPModelWrapper interface from model_registry.py.

Installation requirements per model (run what you need):
  MedCLIP:   pip install git+https://github.com/RyanWangZf/MedCLIP.git
  CheXzero:  pip install git+https://github.com/openai/CLIP.git
  BioViL:    pip install hi-ml-multimodal
  CONCH:     pip install git+https://github.com/mahmoodlab/CONCH.git
  SigLIP:    pip install transformers  (already a dep)
  RAD-DINO:  pip install rad-dino
"""

import torch
from typing import List
from PIL import Image

from .model_registry import CLIPModelWrapper


# ---------------------------------------------------------------------------
# MedCLIP
# ---------------------------------------------------------------------------

class MedCLIPWrapper(CLIPModelWrapper):
    """
    MedCLIP ViT variant.
    Requires: pip install git+https://github.com/RyanWangZf/MedCLIP.git
    Weights:  pretrained/medclip-vit/pytorch_model.bin  (already downloaded)
    Embed dim: 512
    """

    def __init__(self, model_name: str, weights_dir: str = "pretrained/medclip-vit",
                 device: str = "cuda"):
        super().__init__(model_name, device)
        try:
            from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
        except ImportError:
            raise ImportError(
                "MedCLIP package not found.\n"
                "Install: pip install git+https://github.com/RyanWangZf/MedCLIP.git"
            )
        self.processor = MedCLIPProcessor()
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        # Newer transformers drops position_ids from checkpoints; load with strict=False
        import torch.nn as nn
        _orig = nn.Module.load_state_dict
        nn.Module.load_state_dict = (
            lambda self, sd, strict=True, **kw: _orig(self, sd, strict=False, **kw)
        )
        try:
            self.model.from_pretrained(input_dir=weights_dir)
        finally:
            nn.Module.load_state_dict = _orig
        self.model.to(device).eval()

        # MedCLIPFeatureExtractor.convert_rgb was removed in newer transformers;
        # build a manual image transform instead (standard CLIP / ImageNet stats).
        from torchvision import transforms
        self._image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        pixel_values = torch.stack([self._image_transform(img) for img in images]).to(self.device)
        emb = self.model.vision_model(pixel_values)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        # Force max_length=77 (CLIP-style cap). MedCLIP's processor has no
        # model_max_length set, so truncation=True silently no-ops on long
        # captions → position-embedding blow-up downstream.
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        emb = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return emb / emb.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# CheXzero
# ---------------------------------------------------------------------------

class CheXzeroWrapper(CLIPModelWrapper):
    """
    CheXzero: OpenAI CLIP ViT-B/32 fine-tuned on radiology reports.
    Requires: pip install git+https://github.com/openai/CLIP.git
    Weights:  pretrained/chexzero/pytorch_model.bin  (user must download)
    Embed dim: 512

    Download weights from:
      https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno
      Save as: pretrained/chexzero/pytorch_model.bin
    """

    def __init__(self, model_name: str,
                 weights_path: str = "pretrained/chexzero/pytorch_model.bin",
                 device: str = "cuda"):
        super().__init__(model_name, device)
        try:
            import clip
        except ImportError:
            raise ImportError(
                "OpenAI CLIP not found.\n"
                "Install: pip install git+https://github.com/openai/CLIP.git"
            )
        self.clip = clip
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)

        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"CheXzero weights not found at: {weights_path}\n"
                "Download from: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno\n"
                "Save as: pretrained/chexzero/pytorch_model.bin"
            )
        state_dict = torch.load(weights_path, map_location=device)
        # Checkpoint may be raw state dict or wrapped in a dict
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        emb = self.model.encode_image(tensors).float()
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.clip.tokenize(texts, truncate=True).to(self.device)
        emb = self.model.encode_text(tokens).float()
        return emb / emb.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# BioViL (text-only from HF; image via hi-ml-multimodal)
# ---------------------------------------------------------------------------

class BioViLWrapper(CLIPModelWrapper):
    """
    BioViL: CXR-BERT text encoder + BioViL ResNet50 image encoder.
    Text:  transformers AutoModel from microsoft/BiomedVLP-CXR-BERT-specialized
    Image: hi-ml-multimodal package (pip install hi-ml-multimodal)
    Embed dim: 128 (projected)

    Install: pip install hi-ml-multimodal
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        from transformers import AutoTokenizer, AutoModel

        hf_id = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.text_model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(device).eval()

        # Image encoder via hi-ml-multimodal
        # API: health_multimodal.image.model.pretrained.get_biovil_image_encoder
        # Output: ImageModelOutput with .projected_global_embedding [B, 128]
        # Transform requires grayscale PIL (converts to L first)
        try:
            from health_multimodal.image.model.pretrained import get_biovil_image_encoder
            from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
            self.image_model = get_biovil_image_encoder(pretrained=True).to(device).eval()
            self.image_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)
            self._has_image_model = True
        except (ImportError, Exception) as e:
            print(f"[BioViL] Image encoder unavailable ({e}). "
                  "Install hi-ml-multimodal for full support: pip install hi-ml-multimodal")
            self._has_image_model = False

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        if not self._has_image_model:
            raise RuntimeError(
                "BioViL image encoder requires hi-ml-multimodal.\n"
                "Install: pip install hi-ml-multimodal"
            )
        # Transform expects grayscale; convert RGB→L before applying
        tensors = torch.stack([
            self.image_transform(img.convert("L") if img.mode != "L" else img)
            for img in images
        ]).to(self.device)
        out = self.image_model(tensors)
        emb = out.projected_global_embedding  # [B, 128]
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer.batch_encode_plus(
            texts, add_special_tokens=True, padding="longest",
            return_tensors="pt", truncation=True, max_length=256
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        emb = self.text_model.get_projected_text_embeddings(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return emb / emb.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# CONCH (gated — requires HuggingFace access approval)
# ---------------------------------------------------------------------------

class CONCHWrapper(CLIPModelWrapper):
    """
    CONCH: Pathology CLIP (ViT-B-16).
    Requires: pip install git+https://github.com/mahmoodlab/CONCH.git
    GATED model — must request access at: https://huggingface.co/MahmoodLab/conch
    Embed dim: 512
    """

    def __init__(self, model_name: str, hf_token: str = None, device: str = "cuda"):
        super().__init__(model_name, device)
        try:
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        except ImportError:
            raise ImportError(
                "CONCH package not found.\n"
                "Install: pip install git+https://github.com/mahmoodlab/CONCH.git\n"
                "Then request access at: https://huggingface.co/MahmoodLab/conch"
            )
        kwargs = {}
        if hf_token:
            kwargs["hf_auth_token"] = hf_token

        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", **kwargs
        )
        # get_tokenizer() signature varies by CONCH version — try with arg, fall back without
        try:
            self.tokenizer = get_tokenizer("conch_ViT-B-16")
        except TypeError:
            self.tokenizer = get_tokenizer()
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        try:
            emb = self.model.encode_image(tensors, proj_contrast=True, normalize=True)
        except TypeError:
            emb = self.model.encode_image(tensors)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        # CONCH's position-embedding table has a hard limit (typically 128).
        # The CONCH tokenizer does not enforce it, so long captions (e.g. ROCO)
        # blow up the forward pass. Cap at CONTEXT_LEN explicitly.
        CONTEXT_LEN = 127   # 128 - 1 EOS token safety margin
        raw = self.tokenizer(texts)
        # BatchEncoding: extract input_ids and convert to tensor
        if hasattr(raw, "input_ids"):
            ids = raw["input_ids"]
            if not isinstance(ids, torch.Tensor):
                # Truncate long sequences AND pad short ones to a common length
                ids = [list(seq)[:CONTEXT_LEN] for seq in ids]
                max_len = max(len(seq) for seq in ids)
                ids = [seq + [0] * (max_len - len(seq)) for seq in ids]
            else:
                if ids.shape[1] > CONTEXT_LEN:
                    ids = ids[:, :CONTEXT_LEN]
            tokens = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids)
        elif isinstance(raw, list):
            tokens = torch.stack(raw) if isinstance(raw[0], torch.Tensor) else torch.tensor(raw)
            if tokens.dim() == 2 and tokens.shape[1] > CONTEXT_LEN:
                tokens = tokens[:, :CONTEXT_LEN]
        else:
            tokens = raw
        tokens = tokens.to(self.device)
        try:
            emb = self.model.encode_text(tokens, proj_contrast=True, normalize=True)
        except TypeError:
            emb = self.model.encode_text(tokens)
        return emb / emb.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# SigLIP
# ---------------------------------------------------------------------------

class SigLIPWrapper(CLIPModelWrapper):
    """
    SigLIP: Google sigmoid-loss CLIP (ViT-B-16-224).
    Auto-downloads from HuggingFace — no manual steps needed.
    Embed dim: 768
    NOTE: padding="max_length" is required (model was trained this way).
    """

    def __init__(self, model_name: str, model_id: str = "google/siglip-base-patch16-224",
                 device: str = "cuda"):
        super().__init__(model_name, device)
        # Prefer SiglipModel/SiglipProcessor directly — AutoModel/AutoProcessor returns
        # None for SigLIP in some transformers versions.
        try:
            from transformers import SiglipModel, SiglipProcessor
            self.model = SiglipModel.from_pretrained(model_id).to(device).eval()
            self.processor = SiglipProcessor.from_pretrained(model_id)
        except ImportError:
            raise ImportError(
                "SigLIP requires transformers >= 4.38. "
                "Run: pip install transformers --upgrade"
            )

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        # padding="max_length" is required for SigLIP
        inputs = self.processor(
            text=texts, return_tensors="pt",
            padding="max_length", truncation=True
        ).to(self.device)
        emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# RAD-DINO (vision-only)
# ---------------------------------------------------------------------------

class RADDINOWrapper(CLIPModelWrapper):
    """
    RAD-DINO: Microsoft vision-only radiology encoder (ViT-B/14).
    Requires: pip install rad-dino
    Auto-downloads weights from HuggingFace.
    Embed dim: 768 (CLS token)
    NOTE: No text encoder. encode_text() raises NotImplementedError.
         For MedTEC/similarity tasks, use image embeddings only with a separate text model.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        try:
            from rad_dino import RadDino
            self._encoder = RadDino()
            self._use_raw = False
            return
        except ImportError:
            pass
        # Fallback: load directly via transformers
        from transformers import AutoModel
        # AutoImageProcessor was added in transformers 4.25; fall back to AutoFeatureExtractor
        try:
            from transformers import AutoImageProcessor
            self._processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        except ImportError:
            from transformers import AutoFeatureExtractor
            self._processor = AutoFeatureExtractor.from_pretrained("microsoft/rad-dino")
        self._encoder = AutoModel.from_pretrained("microsoft/rad-dino").to(device).eval()
        self._use_raw = True

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        if self._use_raw:
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            outputs = self._encoder(**inputs)
            emb = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            # rad_dino package: processes one image at a time
            cls_list = []
            for img in images:
                cls_emb, _ = self._encoder.extract_features(img)
                cls_list.append(cls_emb.squeeze(0))
            emb = torch.stack(cls_list).to(self.device)
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError(
            "RAD-DINO is a vision-only model and has no text encoder."
        )
