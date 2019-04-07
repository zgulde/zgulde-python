import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='zgulde',
    version='0.0.10',
    author='Zach Gulde',
    author_email='zachgulde@gmail.com',
    description='A Small Person Utility Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/zgulde/zgulde-python',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
