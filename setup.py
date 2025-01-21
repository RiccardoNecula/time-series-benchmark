from setuptools import setup, find_packages

with open("time-series-benchmark/README.md", "r") as f:
    long_description = f.read()

setup(
    name="time series benchmark",
    version="0.1.0",
    description="Python  library for notebooks of time series analysis.",
    author="Riccardo Necula",
    author_email="287627@studenti.unimore.it",
    url="https://github.com/time-series-benchmark",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",

    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "matplotlib",
        "protobuf",
        # Altri requisiti
    ],
    python_requires=">=3.8",
)
