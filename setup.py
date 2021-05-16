from distutils.core import setup
from os.path import exists
from setuptools import setup, find_packages

setup(
    name='credit_scoring',
    version='0.1.0',
    license='MIT',
    description='This library help researcher have quick result with woe, iv, '
                'result reporting with credit scoring problem',
    # long_description=open('README.md').read() if exists("README.md") else "",
    author='Minh Tran',
    author_email='minhtc.uet@gmail.com',
    url='https://github.com/minhtcuet/creditscoring',
    download_url='https://github.com/minhtcuet/creditscoring/archive/v_01.tar.gz',
    keywords=['Credit', 'Credit Score', 'WOE', 'IV', 'WOE-IV', 'GINI', 'KS', 'Lift', 'GAIN'],
    install_requires=[
        'numpy',
        'pandas',
        'plotly',
        'sklearn',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(exclude=['build', 'docs', 'templates']),

)