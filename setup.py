from setuptools import setup, find_packages

setup(
    name="BiTSplitter",                   # Package name
    version="0.1.0",                      # Initial release version
    description="A package for BiTSplitter functionality",  # Short description
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),  # Automatically find packages in ./src
    package_dir={"": "src"},              # Tell distutils that packages are under ./src
    install_requires=[                    # List your package dependencies here
        # 'numpy',
        # 'pandas',
    ],
    entry_points={                        # Optional: create command line executables
        "console_scripts": [
            # For example, if you have a main() in a module "main.py" inside your package:
            # "bitts = your_package.main:main",
        ],
    },
    classifiers=[                         # Additional metadata
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
