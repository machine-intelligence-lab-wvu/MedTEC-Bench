"""
Unified model registry and loader for all CLIP/VLM models.
Provides a consistent interface for encoding images and text across different model families.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from PIL import Image


class CLIPModelWrapper(ABC):
    """Abstract base class for all CLIP-like model wrappers."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode images to normalized embeddings. Returns (N, D) tensor."""
        pass

    @abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to normalized embeddings. Returns (N, D) tensor."""
        pass

    def compute_similarity(self, images: List[Image.Image], texts: List[str]) -> torch.Tensor:
        """Compute cosine similarity matrix (N_images, N_texts)."""
        with torch.no_grad():
            img_emb = self.encode_image(images)
            txt_emb = self.encode_text(texts)
            return img_emb @ txt_emb.T


class OpenCLIPWrapper(CLIPModelWrapper):
    """Wrapper for OpenCLIP-based models (BiomedCLIP, OpenCLIP, PubMedCLIP)."""

    def __init__(self, model_name: str, model_id: str, pretrained: str, device: str = "cuda"):
        super().__init__(model_name, device)
        import open_clip
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_id, pretrained=pretrained, device=device
            )
        except RuntimeError as e:
            if "key(s) in state_dict" in str(e):
                # Known version mismatch (e.g. BiomedCLIP + newer transformers drops
                # position_ids from checkpoints). Retry with strict=False.
                import torch.nn as nn
                _orig = nn.Module.load_state_dict
                nn.Module.load_state_dict = (
                    lambda self, sd, strict=True, **kw: _orig(self, sd, strict=False, **kw)
                )
                try:
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        model_id, pretrained=pretrained, device=device
                    )
                finally:
                    nn.Module.load_state_dict = _orig
            else:
                raise
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        emb = self.model.encode_image(tensors)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(tokens)
        return emb / emb.norm(dim=-1, keepdim=True)


class HFCLIPWrapper(CLIPModelWrapper):
    """Wrapper for HuggingFace transformers CLIP models."""

    def __init__(self, model_name: str, model_id: str, device: str = "cuda"):
        super().__init__(model_name, device)
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        # Force max_length=77 (CLIP's position-embedding table size). Some CLIP
        # processors (e.g. PubMedCLIP) don't declare model_max_length, so
        # `truncation=True` silently no-ops and long captions blow up the
        # position embeddings. Explicit max_length fixes it.
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)


# Modality keys used in MedTEC-Bench
ALL_MODALITIES = ["cxr", "brain_mri", "abdomen_ct", "fundus", "dermoscopy", "histopath"]

# Model registry: all models that can be loaded
MODEL_REGISTRY: Dict[str, dict] = {
    "biomedclip": {
        "class": "OpenCLIPWrapper",
        "model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "pretrained": "",
        "display_name": "BiomedCLIP",
        "domain": "Biomedical (broad)",
        "source": "Microsoft",
        "modalities": ALL_MODALITIES,
    },
    "pubmedclip": {
        "class": "HFCLIPWrapper",
        "model_id": "flaviagiammarino/pubmed-clip-vit-base-patch32",
        "display_name": "PubMedCLIP",
        "domain": "Biomedical (broad)",
        "source": "Flavia G.",
        "modalities": ALL_MODALITIES,
    },
    "medclip": {
        "class": "custom",
        "model_id": "pretrained/medclip-vit",
        "display_name": "MedCLIP",
        "domain": "Medical (broad)",
        "source": "Flax Community",
        "modalities": ALL_MODALITIES,
    },
    "chexzero": {
        "class": "custom",
        "model_id": "chexzero",
        "display_name": "CheXzero",
        "domain": "Radiology (CXR)",
        "source": "Tiu et al.",
        "modalities": ["cxr"],
    },
    "biovil": {
        "class": "custom",
        "model_id": "microsoft/BiomedVLP-CXR-BERT-specialized",
        "display_name": "BioViL",
        "domain": "Radiology (CXR)",
        "source": "Microsoft (Hi-ML)",
        "modalities": ["cxr"],
    },
    "pmc_clip": {
        "class": "OpenCLIPWrapper",
        "model_id": "hf-hub:ryanyip7777/pmc_vit_l_14",
        "pretrained": "",
        "display_name": "PMC-CLIP",
        "domain": "Biomedical papers",
        "source": "PMC-OA team",
        "modalities": ALL_MODALITIES,
    },
    "conch": {
        "class": "custom",
        "model_id": "conch",
        "display_name": "CONCH",
        "domain": "Pathology",
        "source": "Chen et al.",
        "modalities": ["histopath"],
    },
    "plip": {
        "class": "HFCLIPWrapper",
        "model_id": "vinid/plip",
        "display_name": "PLIP",
        "domain": "Pathology",
        "source": "Huang et al.",
        "modalities": ["histopath"],
    },
    "openclip_vitb32": {
        "class": "OpenCLIPWrapper",
        "model_id": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "display_name": "OpenCLIP ViT-B-32",
        "domain": "General",
        "source": "LAION",
        "modalities": ALL_MODALITIES,
    },
    "openclip_vitl14": {
        "class": "OpenCLIPWrapper",
        "model_id": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "display_name": "OpenCLIP ViT-L-14",
        "domain": "General",
        "source": "LAION",
        "modalities": ALL_MODALITIES,
    },
    "siglip": {
        "class": "custom",
        "model_id": "google/siglip-base-patch16-224",
        "display_name": "SigLIP",
        "domain": "General",
        "source": "Google",
        "modalities": ALL_MODALITIES,
    },
    "rad_dino": {
        "class": "custom",
        "model_id": "microsoft/rad-dino",
        "display_name": "RAD-DINO",
        "domain": "Radiology",
        "source": "Microsoft",
        "modalities": [],  # Vision-only, no text encoder
        "note": "Vision-only, no text encoder",
    },
}


def load_model(model_key: str, device: str = "cuda", **kwargs) -> CLIPModelWrapper:
    """Load a model by its registry key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_key]

    if config["class"] == "OpenCLIPWrapper":
        return OpenCLIPWrapper(
            model_name=config["display_name"],
            model_id=config["model_id"],
            pretrained=config.get("pretrained", ""),
            device=device,
        )
    elif config["class"] == "HFCLIPWrapper":
        return HFCLIPWrapper(
            model_name=config["display_name"],
            model_id=config["model_id"],
            device=device,
        )
    elif config["class"] == "custom":
        from .custom_loaders import (
            MedCLIPWrapper, CheXzeroWrapper, BioViLWrapper,
            CONCHWrapper, SigLIPWrapper, RADDINOWrapper,
        )
        custom_map = {
            "medclip": MedCLIPWrapper,
            "chexzero": CheXzeroWrapper,
            "biovil": BioViLWrapper,
            "conch": CONCHWrapper,
            "siglip": SigLIPWrapper,
            "rad_dino": RADDINOWrapper,
        }
        if model_key not in custom_map:
            raise NotImplementedError(f"No custom loader registered for '{model_key}'.")
        return custom_map[model_key](
            model_name=config["display_name"], device=device, **kwargs
        )
    else:
        raise ValueError(f"Unknown class type '{config['class']}' for model '{model_key}'")


def list_models() -> List[str]:
    """List all available model keys."""
    return list(MODEL_REGISTRY.keys())
