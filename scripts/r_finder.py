# -*- coding: utf-8 -*-
"""
R-Finder
========

Script for identifying and assigning R-Groups from journals in pdf format.

author: Jonas Dietrich
email: jonas.dietrich@uni-jena.de

"""

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

import re
import json
import pandas as pd
import numpy as np

from collections import OrderedDict

import pytesseract as tess
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib as mpl

from decimer_segmentation import get_mrcnn_results
sys.path.append("/Users/jonasdietrich/University/DECIMER/R-Group-Detection/DECIMER_R_Model_Jonas")
from Predictor_exported_R import predict_SMILES

from typing import List, Tuple, Dict
from rich.progress import track
from rich import print
import pyfiglet

from decimer_connection import decimerConnection
from decimer_connection import connectRGroup2Structure
from decimer_connection import getCenterDecimerMatch

from predictor_connection import extend2RGBA

def pdf2tiff(path: str, filename: str) -> List:
    """Convert .pdf to .tiff."""

    paper_as_tiff = convert_from_path(path + "/" + filename, fmt="TIFF")

    return paper_as_tiff



def getMetadata4tiff(paper_as_tiff: List) -> Dict:
    """Creates a dictionary with meta data for each page."""

    metadata = pd.DataFrame(columns=["level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text"])

    for i, page in enumerate(track(paper_as_tiff, description="Load pages...")):
        ocr_result = tess.image_to_data(page, lang="eng_chem0.708_105095", output_type="data.frame")
        ocr_result["page_num"] = [i+1] * len(ocr_result["page_num"])
        ocr_result_del_nan = ocr_result.dropna(subset=["text"])
        ocr_result_with_text = ocr_result_del_nan.loc[ocr_result_del_nan["text"] != " "]
        ocr_result_with_text = ocr_result_with_text.loc[ocr_result_del_nan["text"] != "  "]
        ocr_result_with_text = ocr_result_with_text.loc[ocr_result_del_nan["text"] != "   "]
        ocr_result_with_text = ocr_result_with_text.loc[ocr_result_del_nan["text"] != "    "]
        ocr_result_with_text["text"] = list(map(lambda x: x.replace(" ", ""), ocr_result_with_text["text"]))
        next_page = ocr_result_del_nan.loc[ocr_result_del_nan["text"] != " "] # new, no space in join needed anymore
        metadata = pd.concat([metadata, ocr_result_with_text], axis=0, ignore_index=True)
        
    metadata["text"] = metadata["text"].apply(utf2ascii)
    

    return metadata



def utf2ascii(unicode_string: str) -> str:
    """convert utf-8 characters to ascii characters."""

    utf8_ascii_table = {
            b'\xe2\x82\x80':"0",
            b'\xe2\x82\x81':"1",
            b'\xe2\x82\x82':"2",
            b'\xe2\x82\x83':"3",
            b'\xe2\x82\x84':"4",
            b'\xe2\x82\x85':"5",
            b'\xe2\x82\x86':"6",
            b'\xe2\x82\x87':"7",
            b'\xe2\x82\x88':"8",
            b'\xe2\x82\x89':"9",

            b'\xc2\xba':"0",
            b'\xc2\xb9':"1",
            b'\xc2\xb2':"2",
            b'\xc2\xb3':"3",
            b'\xe2\x81\xb4':"4",
            b'\xe2\x81\xb5':"5",
            b'\xe2\x81\xb6':"6",
            b'\xe2\x81\xb7':"7",
            b'\xe2\x81\xb8':"8",
            b'\xe2\x81\xb9':"9",

            b'\xe2\x81\xbb':"-",
            b'\xe2\x81\xba':"+"
            }

    for char in unicode_string:
        utf8_char = char.encode("utf-8")

        for utf8_from_dict, ascii_num in utf8_ascii_table.items():

            if utf8_char == utf8_from_dict:
                unicode_string = unicode_string.replace(char, ascii_num)

    return unicode_string



def REGEXmatching(ocr_text, regex_R_Group):
    matched_by_regex = regex_R_Group.finditer(ocr_text)

    match_strings = []
    match_locations = []
    match_in_text = []
    for match in matched_by_regex:
        match_strings.append(match.groups())
        match_locations.append(match.span())
        start = match.span()[0]
        end = match.span()[1]
        print(ocr_text[start:end])
        match_in_text.append(ocr_text[start:end])

    return match_strings, match_locations, match_in_text

def processREGEXMatch(match_strings, match_in_text):
    
    R_signs_ = []
    R_groups_ = []
    groups_ = []
    for txt in match_in_text:
        groups = re.split("=|:", txt) # no spaces anymore
        R_groups_.append(groups[-1])
        R_signs_.append(groups[:-1])

    #print(groups)
    #print(R_groups_)
    #print(R_signs_)
    return groups_, R_signs_, R_groups_


def fromLocation2Index(text):

    ind_loc_tab = {}
    location = 0
    for n_word, word in enumerate(text):
        catched_location = []
        for char in word:
            catched_location.append(location)
            location += 1
        

        ind_loc_tab[n_word] = catched_location.copy()
        location += 1

    return ind_loc_tab



def isMatch(metadata, match_locations, index2location_table):
    is_match = False
    true_false_values = []
    n_match = []
    match_index = []
    for start_loc, end_loc in match_locations:
        regex_span = list(range(start_loc, end_loc))
        match_index.append(regex_span)
    
    for index_word, location_char in index2location_table.items():
        n_match_list = 0
        for loc in location_char:
            for i in range(len(match_index)):
                if loc in match_index[i]:
                    is_match = True
                    n_match_list = i


        
        true_false_values.append(is_match)
        n_match.append(n_match_list)
        is_match = False

    metadata["is_match"] = true_false_values
    metadata["n_match"] = n_match

    return metadata



def getIndexMatches(match_locations, index2location_table):
    true_false_values = []
    index_matches = []
    for start_loc, end_loc in match_locations:
        middle_loc = int((start_loc + end_loc) / 2) 
        for index_word, location_char in index2location_table.items():
            if middle_loc in location_char:
                index_matches.append(index_word)
                true_false_values.append(True)
            else:
                true_false_values.append(False)

    return index_matches, true_false_values

def getRCoordinates2(match_locations, x_coord, y_coord, words):
    
    match_coord = list(map(lambda i: (x_coord[i], y_coord[i]), match_locations))
    match = list(map(lambda i: words[i], match_locations)) # test it these coordinates are correct

    return match_coord, match



def filterInvalidMatches(match_strings, match_locations):
    """
    This function is only needed if there are mistakes done by tesseract-ocr.
    It filters invalid R-Group matches, which are defined in a list in this function. It is mainly about R=R which is sometimes catched by reading the R-Groups in the structure as R-Group definition, while the equal sign is wrong translated from tesseract-ocr
    """
    invalid_matches = ["R=R", "R:R", "R = R", "R : R", "R = etc."] # replace it with regex
    index_of_invalid_matches = []

    for index, match in enumerate(match_strings):
        if match in invalid_matches:
            index_of_invalid_matches.append(index)

    for index in sorted(index_of_invalid_matches, reverse=True):
        del match_strings[index]
        del match_locations[index]

    
    return match_strings, match_locations

    

def str2smiles(R_replacements):
    """sdfsf"""

    with open("../data/string_smiles.json", "r") as readfile:
        string_smiles_dict = json.load(readfile)

    R_SMILES = []
    r_as_smiles = ""
    for R_group in R_replacements:
        for string, smiles in string_smiles_dict.items():

            if string == R_group:
                r_as_smiles = smiles

            else:
                r_as_smiles = R_group
            
        R_SMILES.append(r_as_smiles)

    #    return r_as_smiles
    return R_SMILES


def insertRinSMILES(R_signs_, R_groups_, S_R_pairs, blabla):
    """Wow"""

    new_smiles = []
    for pair in S_R_pairs:
        num_R_group = pair[0] 
        num_structure = pair[1] 

        predicted_SMILES = blabla[num_structure]

        R_signs_ = [item for sublist in R_signs_ for item in sublist if item != " "]


        new_R_sign = R_signs_[num_R_group].replace(" ", "")
        SMILES_with_R = predicted_SMILES.replace(new_R_sign, R_groups_[num_R_group])

        new_smiles.append(SMILES_with_R) 

    return new_smiles


def showMatches(centerPoints_line, tiff, centerPoints_structures=None, bboxes=None):
    """plot the r."""
    
    fig = plt.figure(figsize=(8.27,11.69))
    ax = fig.add_subplot(111)
    ax.imshow(tiff)

    if type(bboxes) == np.ndarray:
        S_R_pairs = connectRGroup2Structure(centerPoints_structures, centerPoints_line)
        for pair in S_R_pairs:

            ## arrows:
            structure_coord = centerPoints_structures[pair[1]]
            RGroup_coord = centerPoints_line[pair[0]]
            dx = structure_coord[1] - RGroup_coord[0]
            dy = structure_coord[0] - RGroup_coord[1]

            ax.arrow(RGroup_coord[0], RGroup_coord[1], dx=dx, dy=dy, head_width=10, alpha=0.3, color="green", head_starts_at_zero=True)


        for c in centerPoints_structures:

            # centerPoint
            ax.scatter(c[1], c[0], s=400, c="orange", alpha=0.3)


    for r in centerPoints_line:

        last_char_x_coord = r[0]
        last_char_y_coord = r[1]

        ax.scatter(last_char_x_coord,
                last_char_y_coord,
                s=100, c="greenyellow", alpha=0.5)


    plt.axis("off")
    plt.show()


def getPredictedSMILES(all_segments):
    prediis = []
    for i, segments in enumerate(all_segments):
        images = extend2RGBA(segments)

        predictions_SMI = []
        for img in images:
            pred_SMI = predict_SMILES(img)
            predictions_SMI.append(pred_SMI)

        prediis.append(predictions_SMI)
    return prediis

def createMetadata(image_metadata, grouped_metadata):
    metadatas_ = []
    for n_page, center in enumerate(image_metadata["center_point"]):
        if center != [] and (n_page+1) in list((grouped_metadata["page_num"]).astype(int)):
            text_metadata = grouped_metadata.loc[grouped_metadata["page_num"] == str(n_page+1)].reset_index(drop=True)
            x_coord = text_metadata.left.astype(float)
            y_coord = text_metadata.top.astype(float)
            match_coord = [(x,y) for x,y in zip(x_coord, y_coord)]
            
            S_R_pairs = connectRGroup2Structure(center, match_coord)
            
            text_metadata["image_ref"] = ""
            text_metadata["center_image"] = ""
            for R, S in S_R_pairs:
                text_metadata["image_ref"][R] = S
                text_metadata["center_image"][R] = center[S]
            metadatas_.append(text_metadata.copy())

        if center == [] and n_page in grouped_metadata["page_num"]:
            print("no bboxes found")
        if center != [] and n_page not in grouped_metadata["page_num"]:
            print("no R_group found") # important for tables on the other side
    return metadatas_


def getMolSMILES(image_metadata, metadatas_, page, i):
    pred_smiles_ = image_metadata["pred_SMILES"][page-1]
    try: 
        for i9, ref in enumerate(metadatas_[i]["image_ref"]):
            print("-+++++++++++++++++++++++++++-")
            #print(f"[{metadatas_[i]['R_signs'][i9][0]}]")
            #print(metadatas_[i]["R_SMILES"][i9])
            for R_sign in metadatas_[i]["R_signs"][i9]:
                print("-------------------------")
                print(pred_smiles_[ref])
                print(R_sign)
                pred_smiles_[ref] = pred_smiles_[ref].replace(f"[{R_sign}]", metadatas_[i]["R_SMILES"][i9]) # Achtung hier überschreibst du etwas
            print(pred_smiles_[ref])
            print("-------------------------")
        return pred_smiles_
    except IndexError:
        print("hier fehler!!!!!!!!!!!!!!!!!!!!!!")
        return None

def createImageMetadata(paper_as_tiff, all_bboxes, all_segments, prediis, centerPoints_structures):
        image_metadata = pd.DataFrame({"tiff" : paper_as_tiff,
                                       "bboxes" : all_bboxes,
                                       "segments" : all_segments
                                         }) 
        image_metadata["pred_SMILES"] = prediis 
        image_metadata["center_point"] = centerPoints_structures 
        return image_metadata

def get_REGEX_matches(metadata, R_groups_, R_signs_):
    matches_metadata = metadata.loc[metadata["is_match"] == True]
    matches_metadata = matches_metadata.applymap(str)
    grouped_metadata = matches_metadata.groupby("n_match", as_index=False, sort=False).agg({"level" : "first", "page_num": "first", "block_num": "first", "par_num": "first", "line_num": "first", "word_num": "first", "left": "first", "top": "first", "width": "first", "height": "first", "conf": "first", "text": "first", "is_match": "first"})
    R_SMILES = str2smiles(R_groups_)
    grouped_metadata["R_signs"] = R_signs_
    grouped_metadata["R_SMILES"] = R_SMILES
    return grouped_metadata

def getImageData(image_metadata, metadatas_, page, i):
    tiff = image_metadata["tiff"][page-1]
    x_coord = metadatas_[i].left.astype(float)
    y_coord = metadatas_[i].top.astype(float)
    match_coord = [(x,y) for x,y in zip(x_coord, y_coord)]
    centerPoints_structures = list(metadatas_[i]["center_image"])  
    bboxes = image_metadata["bboxes"][page]
    return tiff, match_coord, centerPoints_structures, bboxes
    

def main():

    regex_R_Group = re.compile("\d*(?<=[\d ,])(\ *[RXY][0-9 '’]*,?)+[=:]\ *([^\s,;:]+)(((?=,|\ and|\ or),|\ and|\ or)\ ?(?![RXY])[^\s,;:]+)*")
    regex_R_Group = re.compile("(?<=[\d ,])([RXY][0-9 '’]*,?)+[=:]\ *([^\s,;.:]+)(((?=,|\ and|\ or),|\ and|\ or)\ ?(?![RXY])[^\s,;.:]+)*") # without number at the beginning and delete space at the beginning
    
    paper_as_tiff = pdf2tiff(path="../data/PHYTOCH/", filename="1-s2.0-S1874390018304543-main.pdf")
    metadata = getMetadata4tiff(paper_as_tiff)

    ## function calls
    text_in_list = metadata.text.to_list()
    text_in_string = " ".join(text_in_list) # change sentences to text
    match_strings, match_locations, match_text = REGEXmatching(text_in_string, regex_R_Group)
    match_strings, match_locations = filterInvalidMatches(match_strings, match_locations) # error correction
    groups_, R_signs_, R_groups_ = processREGEXMatch(match_strings, match_text)
    ind_loc_tab = fromLocation2Index(text_in_list)

    metadata = isMatch(metadata, match_locations, ind_loc_tab) 
    grouped_metadata = get_REGEX_matches(metadata, R_groups_, R_signs_)
    print(grouped_metadata)

    # call decimer
    all_bboxes, all_segments = decimerConnection(paper_as_tiff) 
    centerPoints_structures = getCenterDecimerMatch(all_bboxes) 
    prediis = getPredictedSMILES(all_segments)

    image_metadata = createImageMetadata(paper_as_tiff, all_bboxes, all_segments, prediis, centerPoints_structures)
    metadatas_ = createMetadata(image_metadata, grouped_metadata)

    page_num_with_match = set((grouped_metadata["page_num"]).astype(int))
    for i, page in enumerate(page_num_with_match):
        print(i)
        pred_smiles_ = getMolSMILES(image_metadata, metadatas_, page, i)
        if pred_smiles_ is not None:

            tiff, match_coord, centerPoints_structures, bboxes = getImageData(image_metadata, metadatas_, page, i) 
            showMatches(match_coord, tiff, centerPoints_structures, bboxes)


if __name__ == "__main__":

    title = pyfiglet.figlet_format("R=finder", font = "speed")
    print(f'[green]{title}[/green]')

    main()

    #MDPI = ["molecules-05-01429.pdf", "molecules-12-01910.pdf", "molecules-14-02888.pdf", "molecules-17-06317.pdf", "molecules-20-16852.pdf", "molecules-08-00053.pdf", "molecules-14-02016.pdf", "molecules-14-03780.pdf", "molecules-18-06620.pdf", "molecules-21-00748-v2.pdf", "molecules-12-01153.pdf", "molecules-14-02202.pdf", "molecules-15-07313.pdf", "molecules-19-19610.pdf", "molecules-21-00901.pdf", "molecules-12-01679.pdf", "molecules-14-02373.pdf", "molecules-15-09057.pdf", "molecules-20-03898.pdf", "molecules-23-00134-v2.pdf"]
    #for filename in MDPI[8:]:
        #paper_as_tiff = pdf2tiff(path="../data/MDPI/", filename=filename)            
        #print("*************")
        #print(filename)
        #print("*************")

        #JNPRDF = ["acs.jnatprod.0c00968.pdf", "acs.jnatprod.1c00980.pdf", "acs.jnatprod.9b01284.pdf", "acs.jnatprod.0c01340.pdf", "acs.jnatprod.1c01024.pdf", "synthesis/acs.jnatprod.0c00310.pdf", "synthesis/acs.jnatprod.6b00510.pdf", "synthesis/np049902u.pdf", "synthesis/acs.jnatprod.0c00697.pdf", "synthesis/acs.jnatprod.7b00311.pdf", "synthesis/np060219c.pdf", "synthesis/acs.jnatprod.0c01317.pdf", "synthesis/acs.jnatprod.7b00405.pdf", "synthesis/np300790g.pdf", "synthesis/acs.jnatprod.5b01041.pdf", "synthesis/acs.jnatprod.8b00215.pdf", "synthesis/np960169j.pdf", "synthesis/acs.jnatprod.6b00350.pdf", "synthesis/acs.jnatprod.8b00218.pdf", "synthesis/np970246q.pdf"]
        #for filename in JNPRDF[6:7]: # 5. probleme
        #    paper_as_tiff = pdf2tiff(path="../data/JNPRDF", filename=filename) 


#    PHYTOCH = ["1-s2.0-S1874390018304543-main.pdf", "1-s2.0-S1874390019300941-main.pdf", "synthesis/1-s2.0-S003194220200016X-main.pdf", "synthesis/1-s2.0-S0031942202004752-main.pdf", "synthesis/1-s2.0-S0031942203000037-main.pdf", "synthesis/1-s2.0-S0031942204002882-main.pdf", "synthesis/1-s2.0-S003194220500110X-main.pdf", "synthesis/1-s2.0-S0031942205001627-main.pdf", "synthesis/1-s2.0-S0031942209001678-main.pdf", "synthesis/1-s2.0-S0031942210000567-main.pdf", "synthesis/1-s2.0-S0031942215001727-main.pdf", "synthesis/1-s2.0-S0031942216302734-main.pdf", "synthesis/1-s2.0-S0031942217304181-main.pdf", "synthesis/1-s2.0-S0031942219305485-main.pdf", "synthesis/1-s2.0-S0031942220303393-main.pdf", "synthesis/1-s2.0-S0031942220303538-main.pdf", "synthesis/1-s2.0-S0031942221002363-main.pdf", "synthesis/1-s2.0-S0031942221002880-main.pdf", "synthesis/1-s2.0-S003194229900494X-main.pdf", "synthesis/1-s2.0-S0031942299006184-main.pdf"]
#    for filename in PHYTOCH:
#        print("------------------")
#        print(filename)
#        print("------------------")
