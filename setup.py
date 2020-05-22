from pathlib import Path
from setuptools import setup, find_packages


def read_requirements():
    '''parses requirements from requirements.txt'''
    reqs_path = Path(__file__).parent / 'requirements.txt'
    reqs = None
    with open(reqs_path) as reqs_file:
        reqs = reqs_file.read().splitlines()
    return reqs


readme_path = Path(__file__).parent / 'README.md'
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='covid_sicr',
    version='1.0.3',
    author='NIDDK',
    packages=find_packages(),
    license='MIT',
    description='Global estimation of unobserved COVID-19 infection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=read_requirements(),
    )
