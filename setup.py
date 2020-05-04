from setuptools import setup

import versioneer

version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()

setup(
    name="attrici",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="Counterfactual climate data from past datasets for the ISIMIP project.",
    url="http://github.com/isi-mip/attrici",
    author="Matthias Mengel, Simon Treu",
    author_email="matthias.mengel@pik-potsdam.de",
    license="GPLv3",
    packages=["attrici"],
    zip_safe=False,
)
