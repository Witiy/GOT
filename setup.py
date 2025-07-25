from setuptools import setup, find_packages, Extension
setup(
    name='py-scgot',
    version='0.3.0',
    author='Ruihong Xu',
    author_email='xuruihong@big.ac.cn',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    install_requires=[
        "cellrank==2.0.4",
        "joblib==1.3.2",
        "matplotlib",
        "networkx>=3.1",
        "numpy==1.26.3",
        "pygam>=0.9.0",
        "numba==0.58.1",
        "llvmlite==0.41.1",
        "POT>=0.9.3",
        "scanpy==1.9.3",
        "scikit-learn==1.1.3",
        "scipy==1.11.4",
        "scvelo>=0.3.0",
        "seaborn",
        "torchdiffeq==0.2.4",
        "tqdm",
        "plotly",
        "setuptools",
        "statsmodels",
],


)
