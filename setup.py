import os

from setuptools import setup, find_packages

base_dir = os.path.dirname(os.path.abspath(__file__))


def get_long_description():
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read()


def get_project_version():
    version = {}
    with open(os.path.join(base_dir, "tlxzoo", "version.py"), encoding="utf-8") as fp:
        exec(fp.read(), version)
    return version


version = get_project_version()


def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename)) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")


setup(
    name="tlxzoo",
    version=version["__version__"],
    license="apache",
    description="zoo for tensorlayerx",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="tensorlayerx",
    author_email="",
    url="",
    keywords="tensorlayerx zoo",
    python_requires=">=3.5",
    install_requires=install_requires,
)
