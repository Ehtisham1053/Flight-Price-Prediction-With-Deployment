# the main function of the setup file is to make the whole application as the package 
# the folder that contain the __init__ file it setup consider it as the package
from setuptools import find_packages , setup
from typing import List

hype = "-e ."
def get_requirement(file_path:str) ->List[str]:
    '''
    this is the function to get all the requirement from the txt file and install
    '''

    requirement=[]
    with open (file_path) as file_p:
        requirement=file_p.readline()
        requirement = [req.replace("\n" , "") for req in requirement] 

    if hype in requirement:
        requirement.remove(hype)

    return requirement





setup(
    name="Flight price prediction using machine learning",
    version="1.0",
    author="Ehtisham Afzal",
    author_email="2020n08248@gmail.com",
    packages=find_packages(),
    install_require=get_requirement("requirement.txt")  # from the file get all the packages to install 
)







