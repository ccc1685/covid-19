from pathlib import Path

try:
    from pip.req import parse_requirements
    from pip.download import PipSession
except ImportError:
    from pip._internal.req import parse_requirements
    try:
        from pip._internal.download import PipSession
    except ImportError:
        from pip._internal.network.session import PipSession

from setuptools import setup, find_packages


def read_requirements():
    '''parses requirements from requirements.txt'''
    reqs_path = Path(__file__).parent / 'requirements.txt'
    install_reqs = parse_requirements(str(reqs_path), session=PipSession())
    reqs = [str(ir.req) for ir in install_reqs]
    return reqs

readme_path = Path(__file__).parent / 'README.md'
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='covid_sicr',
    version='1.0.2',
    author='NIDDK',
    packages=find_packages(),
    license='MIT',
    description='Global estimation of unobserved COVID-19 infection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=read_requirements(),
    )
