from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Define Cython extensions
extensions = [Extension("core_utils.cython_transformers", ["core_utils/cython_transformers.pyx"])]

setup(
    name="lylexpy",
    version="0.1.0",
    description="Lylex and WDBX application",
    packages=find_packages(exclude=[".venv", "tests", "docs", "plugins"]),
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "lylex-cli = lylex.cli:cli",
            "wdbx-cli = wdbx.cli:cli",
        ],
        "myapp.plugins": [],
        "myapp.async_plugins": [],
        "wdbx.plugins": [],
        "wdbx.async_plugins": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    extras_require={
        "dev": [
            "cython>=0.29.0",
            "mypy",
            "flake8",
            "black",
            "isort",
            "pydocstyle",
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "pytest-asyncio",
            "pytest-timeout",
            "pytest-mock",
            "pytest-asyncio",
        ],
    },
    ext_modules=cythonize(extensions),
)
