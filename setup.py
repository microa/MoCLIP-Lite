"""
Setup script for MoCLIP-Lite package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="moclip-lite",
    version="1.0.0",
    author="Binhua Huang, Ni Wang, Arjun Pakrashi, Soumyabrata Dev",
    author_email="",
    description="Efficient Video Recognition by Fusing CLIP with Motion Vectors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microa/MoCLIP-Lite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "moclip-train-mv=training.train_mv_tsn:main",
            "moclip-train-fusion=training.train_fusion:main",
            "moclip-evaluate=evaluation.evaluate_zeroshot:main",
        ],
    },
)
