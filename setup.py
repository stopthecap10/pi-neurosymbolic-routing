"""Setup script for pi-neuro-routing package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="pi-neuro-routing",
    version="1.0.0",
    description="Neurosymbolic routing for LLMs on edge devices (Raspberry Pi 4B)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Avyay Sadhu",
    author_email="",
    url="https://github.com/avyaysadhu/pi-neurosymbolic-routing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "sympy==1.14.0",
        "pandas==2.3.3",
        "numpy==2.4.0",
        "requests==2.32.5",
        "python-dateutil",
        "pytz",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pi-neuro-run=pi_neuro_routing.cli.run_experiment:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="llm neuro-symbolic edge-computing raspberry-pi phi-2 sympy",
)
