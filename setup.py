from setuptools import find_packages, setup

setup(
    name='helloml',
    packages=find_packages(include=['helloml']),
    version='0.1.0',
    description='Machine learning, simplified',
    author='Christine Yong',
    license='MIT',
    install_requires=[
        'pandas>=1.1.3',
        'numpy>=1.19.2',
        'scikit-learn>=0.23.2',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=4.4.1'],
    test_suite='tests',
)
