from setuptools import setup

url = 'https://github.com/s-omranpour/Social-Phi'
setup(
    name='social_phi',
    version='0.0.1',
    description='a python package to calculate temporal integrated information of a group of people based on their temporal activites as time series',
    url=url,
    author='Soroush Omranpour',
    author_email='soroush.333@gmail.com',
    keywords='information theory, integrated information, consciousness, social network',
    packages=['social_phi'],
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy >= 1.7.0'
    ],
    license="MIT license",

    project_urls={  # Optional
        'Bug Reports': url + '/issues',
        'Source': url,
    },
)