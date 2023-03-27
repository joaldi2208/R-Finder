import re
import json
import pandas as pd
import numpy as np

from PIL import Image

import pytesseract as tess
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib as mpl

from decimer_segmentation import get_mrcnn_results
from decimer_segmentation import apply_masks
from decimer_segmentation import save_images

def decimerConnection(paper_as_tiff):
    """sdfsd"""
    
    all_bboxes = []
    all_segments = []

    for tiff in paper_as_tiff:

        imarray = np.array(tiff)
        masks, bboxes, _ = get_mrcnn_results(imarray)
        segments, _ = apply_masks(imarray, masks) 

        all_bboxes.append(bboxes)
        all_segments.append(segments)
        
     
    return all_bboxes, all_segments



def getCenterDecimerMatch(all_bboxes): # eigentlich nur bboxes mit R drin
    """get the center friend."""
    
    cP = []
    for bboxes in all_bboxes:
        cPoints_ = []
        for coord in bboxes:
            vec1 = coord[:2]
            vec2 = coord[2:]

            centerPoint = 1/2 * (vec1 + vec2)
            cPoints_.append(centerPoint)
        cP.append(cPoints_)

    return cP



def connectRGroup2Structure(loc_structures, loc_RGroups):
    """connect."""

    structure_RGroup_pairs_ = []
    for r_index, r_loc in enumerate(loc_RGroups):
        distances = []
        for s_loc in loc_structures:

            dist = np.sqrt((r_loc[0] - s_loc[1])**2 + (r_loc[1] - s_loc[0])**2)
            distances.append(dist)
        ind_closest_struct = np.argmin(distances)
        structure_RGroup_pairs_.append((r_index, ind_closest_struct))
    
    return structure_RGroup_pairs_

