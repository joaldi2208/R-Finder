import re
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib as mpl 


# C1CCC(CC1)(C2=CC=C(C=C2)O)C3=CC=C(C=C3)O 
# O is OH !!!! I didn't know


def replaceRinSMILES(R_replacements, r_in_smiles):
    
    for OCR_identifier, R_group in R_replacements:

        R_replaced_SMILES = r_in_smiles.replace(OCR_identifier, R_group)
            



r_in_smiles = "C1CCC(CC1)(C2=CC=C(C=C2)R)C3=CC=C(C=C3)R)"

R_replacements = [("R", "MOM"), ("R", "OH"), ("R", "H")]
replaceRinSMILES(R_replacements, r_in_smiles)
        
    
    
