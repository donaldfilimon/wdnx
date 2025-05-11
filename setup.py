from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="scripts_app",
    version="0.1.0",
    description="Lylex and WDBX application",
    packages=find_packages(exclude=[".venv", "tests", "docs", "plugins"]),
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "lylex-cli = cli:cli",
            "wdbx-cli = wdbx_cli:cli",
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
    python_requires=">=3.7",
)
