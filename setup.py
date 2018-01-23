from setuptools import setup

setup(
    name='conrep-text',
    version='0.1',
    packages=['conrep-text'],
    url='https://github.com/BarathiGanesh-HB/conrep-text/',
    download_url = 'https://github.com/BarathiGanesh-HB/conrep-text/archive/0.1.tar.gz',
    license='MIT',
    author='Barathi Ganesh HB',
    author_email='barathiganesh.hb@gmail.com',
    description='Recursive Linear Transformer',
    install_requires=[
        'numpy',
        'nltk',
        'gensim',
    ],
)
