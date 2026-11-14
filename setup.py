from setuptools import find_packages, setup

setup(
    name="sigmeru",
    version="1.0",
    python_requires=">=3.12",
    zip_safe=False,
    packages=find_packages(include=["sigmeru", "sigmeru.*"]),
)