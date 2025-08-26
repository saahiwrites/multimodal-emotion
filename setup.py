from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multimodal-emotion-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-modal emotion detection using transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/multimodal-emotion-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "torchaudio>=0.9.0",
        "torchvision>=0.10.0",
        "librosa>=0.9.0",
        "opencv-python>=4.5.0",
        "pillow>=8.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pyyaml>=6.0",
    ],
)
