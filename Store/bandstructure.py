#### Bandstructure plotting
# pylint: disable = C0103, C0114, C0116, C0301, R0914

import os

def bandstructure_input():
    while True:
        BS_directory = input("Please input the directory of bandstructure calculation: ")
        if os.path.exists(BS_directory):
            print(f"Your bandstructure calculation is located in {BS_directory}.")
            return BS_directory
        else:
            print("The directory does not exist. Please try again.")

