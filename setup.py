from setuptools import find_packages,setup  # Import necessary functions from setuptools for package setup.
from typing import List  # Import List type for type hints.

hyphen_dot_e="-e ."  # Define a string for the editable install flag.
def get_requirements(file_path:str) ->List[str]:  # Define a function to read requirements from a file.
    requirements=[]  # Initialize an empty list for requirements.
    with open(file_path) as temp_file:  # Open the file in read mode.
        requirements=temp_file.readlines()  # Read all lines into the list.
        requirements=[req.replace("\n","") for req in requirements]  # Remove newline characters from each line.

        if hyphen_dot_e in requirements:  # Check if the editable flag is in the list.
            requirements.remove(hyphen_dot_e)  # Remove it if present.

setup(  # Call the setup function to configure the package.
    name="mlprojects",  # Package name.
    version='0.0.1',  # Version number.
    author='Ayush',  # Author name.
    author_email='ayukhanalsh100@gmail.com',  # Author email.
    packages=find_packages(),  # Automatically find packages.
    install_requires=get_requirements('requirements.txt')  # Dependencies from the function.
)