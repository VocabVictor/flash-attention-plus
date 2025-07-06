"""
Setup script for FlashAttention with FlagGems backend
"""

from setuptools import setup, find_packages
import os

# Read the README file
this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="flash-attn-plus",
    version="0.0.1",
    author="Zhongheng Wu",
    author_email="",
    description="Hardware-agnostic FlashAttention implementation using FlagGems/Triton backend",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhonghengwu/flash-attention-plus",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "einops",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-benchmark",
        ],
    },
)