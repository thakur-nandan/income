from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='income',
    version='0.0.2',
    author="Nandan Thakur",
    author_email="nandant@gmail.com",
    description='Domain Adaptation for Memory-Efficient Dense Retrieval',
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url='https://github.com/NThakur20/income',
    download_url="https://github.com/NThakur20/income/archive/v0.0.2.zip",
    packages=find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'torch >= 1.9.0',
        'transformers >= 4.3.3',
        'tensorboard >= 2.5.0',
        'boto3'
    ],
    keywords="Information Retrieval Transformer Networks BERT PyTorch IR NLP deep learning"
)
