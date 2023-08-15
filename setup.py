from setuptools import find_packages,setup   #--> it will find all packages available in ML al
from typing import List


Hyphen_e = '-e .'

def get_requirement(file_path:str)->List[str]:  #--> it wiil take file as str and return list of str
    
    requirement=[]
    with open(file_path) as file_obj:
        requirement= file_obj.readlines()
        requirement=[req.replace("\n", "") for req in requirement]
        
        if Hyphen_e in requirement:
            requirement.remove(Hyphen_e)  #--> we dont want to req detect this

    return requirement



setup(
    name=  'ML_project',
    version= '0.0.1',
    description= 'Ml project practice',
    author= 'vaibhav B',
    author_email= 'vaibhav.b108@gmail.com',
    packages= find_packages(),                    #--> it will cehck file present in __init__.py and 
                                                    #run required library no need to import 
    install_requires=  get_requirement('requirements.txt')  #--> it will go in requirement file and run all req
    ) 