from setuptools import setup

setup(
    name='noise-detector',
    version='0.1.0',
    description='A simple tool to detect changes in environment noise',
    url='https://github.com/matthewscholefield/noise-detector',
    author='Matthew Scholefield',
    author_email='matthew331199@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='noise detection audio',
    py_modules=['noise_detector'],
    install_requires=[
        'pylisten',
        'sonopy',
        'numpy',
        'prettyparse',
        'logzero'
    ],
    entry_points={
        'console_scripts': [
            'noise-detector=noise_detector:main',
        ],
    }
)
