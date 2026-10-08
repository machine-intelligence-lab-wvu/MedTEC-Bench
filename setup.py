from setuptools import setup, find_packages

setup(
    name="medvlm-probe",
    version="0.1.0",
    description=(
        "MedVLM-Probe: Diagnosing Text Embedding Collapse and Spatial "
        "Grounding Failures in Medical Vision-Language Models."
    ),
    author="Anonymous",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "open_clip_torch>=2.20",
        "transformers>=4.30",
        "pillow>=9.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pandas>=2.0",
        "tqdm>=4.65",
        "huggingface_hub>=0.16",
        "nibabel>=5.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
        "notebooks": ["jupyter", "ipywidgets"],
    },
)
