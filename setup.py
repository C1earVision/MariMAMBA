from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mario-level-generator",
    version="0.2.0",
    author="Ahmed Ebrahim Al Mohamady",
    author_email="ahmedebrahim3690@gmail.com",
    description="Conditional Mamba for discrete level generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/placeholder/mario-level-generator",
    packages=find_packages(exclude=["tests", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "viz": [
            "mario-gpt>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mario-gen=scripts.generate_levels:main",
            "mario-train-ae=scripts.train_autoencoder:main",
            "mario-train-mamba=scripts.train_mamba:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mario_level_generator": ["config/*.yaml"],
    },
)