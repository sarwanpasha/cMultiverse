from setuptools import setup, find_packages
from pathlib import Path

readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in (Path(__file__).parent / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="scmultiverse",
    version="0.1.0",
    description=(
        "Sample-efficient specification-curve analysis for single-cell RNA-seq. "
        "Quantifies which analytical choices most determine which biological "
        "conclusions using Sobol sensitivity analysis on a Gaussian-process "
        "surrogate."
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "scmultiverse-run=scmultiverse.run_multiverse:main",
            "scmultiverse-audit=scmultiverse.audit_published_claims:main",
            "scmultiverse-figures=scmultiverse.make_figures:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
)
