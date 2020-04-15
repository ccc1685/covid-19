import os

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
    reqs_path = os.path.join('.', 'requirements.txt')
    install_reqs = parse_requirements(reqs_path, session=PipSession())
    reqs = [str(ir.req) for ir in install_reqs]
    return reqs

setup(
    name='covid',
    version=1.0,
    author='NIDDK',
    packages=find_packages(),
    license='MIT',
    description='Global estimation of unobserved COVID-19 infection',
    install_requires=read_requirements(),
    )