import re
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib as mpl 



def replaceRinSMILES(R_replacements, r_in_smiles):
    "replaces the R-Group in the predicted structures with the found smiles in the R-Group definition"""    
    for OCR_identifier, R_group in R_replacements:

        R_replaced_SMILES = r_in_smiles.replace(OCR_identifier, R_group)
            



if __name__ == "__main__":
    r_in_smiles = "C1CCC(CC1)(C2=CC=C(C=C2)R)C3=CC=C(C=C3)R)"

    R_replacements = [("R", "MOM"), ("R", "OH"), ("R", "H")]
    replaceRinSMILES(R_replacements, r_in_smiles)
            
    
    
