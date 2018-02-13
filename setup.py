from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='decoding_decoders',
    version='0.0.1',
    description='Decoding Decoders: Finding Optimal Representation Spaces'
                ' for Unsupervised Similarity Tasks',
    long_description=readme,
    author='Bablyon AI Research',
    author_email='nils.hammerla@babylonhealth.com',
    url='https://github.com/Babylonpartners/decoding-decoders',
    packages=find_packages()
)
