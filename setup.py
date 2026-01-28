"""Setup script for GPU Optimization Demo."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gpu-optimize-demo",
    version="0.1.0",
    author="GPU Optimization Team",
    author_email="team@example.com",
    description="Enterprise-grade GPU Profiling, IO Optimization & NCCL Communication Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/gpu_optimize_demo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpu-profile=src.profiling.cli:main",
            "io-benchmark=src.io_optimize.cli:main",
            "nccl-analyze=src.nccl.cli:main",
        ],
    },
)
