import re
import argparse
import glob
import logging
import os
import sys
import random
import json
from functools import partial

import pandas as pd
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import re

def rule_based(document_id, text):
    text = text.lower()

    phone_pattern_1 = '\d\d\d-\d\d\d-\d\d\d\d'
    phone_pattern_2= '\(\d\d\d\)-? ?\d\d\d-\d\d\d\d'
    phone_pattern_3= '\(?\d\d\d\)?\.? ?\d\d\d\.\d\d\d\d'

    website_pattern = "((www\.\w+(\.\w+)|((\w + \.)+com)))"
    
    date_pattern1 = '[\n ]\d{4}-[0 1]?\d{1}-[0,1,2,3]?\d'
    date_pattern12 = '[\n ]\d{4}/[0 1]?\d{1}/[0,1,2,3]?\d'
    date_pattern13 = '[\n ]\d{4}\.[0 1]?\d{1}\.[0,1,2,3]?\d'
    
    date_pattern2 =  '[ \n][0 1]?\d{1}-[0 1 2 3]?\d{1}-\d{4}'
    date_pattern22 =  '[ \n][0 1]?\d{1}/[0 1 2 3]?\d{1}/\d{4}'
    date_pattern23 =  '[ \n][0 1]?\d{1}\.[0 1 2 3]?\d{1}\.\d{4}'
    

    
    date_pattern2y =  '[ \n][0 1]?\d{1}-[0 1 2 3]?\d{1}-\d{2}'
    date_pattern22y =  '[ \n][0 1]?\d{1}/[0 1 2 3]?\d{1}/\d{2}'
    date_pattern23y =  '[ \n][0 1]?\d{1}\.[0 1 2 3]?\d{1}\.\d{2}'
    
    patient_id = '\d{6}'
    DaysOfWeeks = ["monday", "tuesday","wednesday","thursday", "friday" ,"mon", "tue", "wed" ,"fri"]
    Months = ["january", "jan", "febuary" ,"feb", "march" ,"mar", "april" ,"apr", "june", "jun", "july" ,"jul"]
    Months2 = ["august" ,"aug" ,"septempber","sept", "october" ,"oct", "november" ,"nov" ,"december"]
    
    additional_rules = DaysOfWeeks + Months +Months2
    additional_rules = ["[ , . : ]"+x+"[ , .]" for x in additional_rules]
    
    
    
    rules = [
        phone_pattern_1,phone_pattern_2,phone_pattern_3,
        website_pattern,
        date_pattern1,date_pattern12,date_pattern13,date_pattern2,date_pattern22,date_pattern23,
        date_pattern2y,date_pattern22y,date_pattern23y,]
    rules.extend(additional_rules)

    
    location = [0]*len(text)
    
    for rule in rules:
        for match in re.finditer(rule, text):
            i, j = match.span()
            location[i:j] = [1]*(j-i)
    location2 = [location[i] if text[i] != ' ' else 0 for i in range(len(text))]
            
    annotations = pd.DataFrame(columns=[
        'document_id', 'annotation_id', 'start', 'stop', 'entity', 'entity_type', 'comment' 
    ])
    is_start = False 
    for i in range(len(location2)):
        if location2[i] == 1 and not is_start:
            start = i 
            is_start = True
        if location2[i] == 0 and is_start:
            assert (start is not None)
            stop = i 
            is_start = False  
            entity = text[start:stop]
            entity_type = 'PHI'
            annotations = annotations.append({
                'document_id': document_id, 'annotation_id':'', 'start': start, 'stop':stop,
                'entity': entity, 'entity_type': entity_type, 'comment':''
            }, ignore_index=True)
            start, stop = None, None

    return annotations

# def rule_based(document_id, text):
#     text = text.lower()

#     phone_pattern_1 = '\d\d\d-\d\d\d-\d\d\d\d'
#     phone_pattern_2= '\(\d\d\d\)-? ?\d\d\d-\d\d\d\d'
#     phone_pattern_3= '\(?\d\d\d\)?\.? ?\d\d\d\.\d\d\d\d'
#     fax_pattern_1 =  "[Ff]ax(ed)?[^/d/n]{0,20}(\(?\d{3}\)?)"
#     fax_pattern_2 = "[- \t]?\d{3}[- \t]\d{4}"
#     website_pattern = "((www\.\w+(\.\w+)|((\w + \.)+com)))"
    
#     date_pattern1 = '[\n ]\d{4}-[0 1]?\d{1}-[0,1,2,3]?\d'
#     date_pattern12 = '[\n ]\d{4}/[0 1]?\d{1}/[0,1,2,3]?\d'
#     date_pattern13 = '[\n ]\d{4}\.[0 1]?\d{1}\.[0,1,2,3]?\d'
    
#     date_pattern2 =  '[ \n][0 1]?\d{1}-[0 1 2 3]?\d{1}-\d{4}'
#     date_pattern22 =  '[ \n][0 1]?\d{1}/[0 1 2 3]?\d{1}/\d{4}'
#     date_pattern23 =  '[ \n][0 1]?\d{1}\.[0 1 2 3]?\d{1}\.\d{4}'
    
#     date_pattern1y = '[\n ]\d{2}-[0 1]?\d{1}-[0,1,2,3]?\d'
#     date_pattern12y = '[\n ]\d{2}/[0 1]?\d{1}/[0,1,2,3]?\d'
#     date_pattern13y = '[\n ]\d{2}\.[0 1]?\d{1}\.[0,1,2,3]?\d'
    
#     date_pattern2y =  '[ \n][0 1]?\d{1}-[0 1 2 3]?\d{1}-\d{2}'
#     date_pattern22y =  '[ \n][0 1]?\d{1}/[0 1 2 3]?\d{1}/\d{2}'
#     date_pattern23y =  '[ \n][0 1]?\d{1}\.[0 1 2 3]?\d{1}\.\d{2}'
    
#     patient_id = '\d{6}'
#     DaysOfWeeks = ["monday ", "tuesday"," wednesday"," thursday", " friday" ," mon ", " tue ", " wed " ," tr " ," fri "]
#     Months = [" january ", " jan ", "febuary" ," feb ", " march " ," mar ", " april " ," apr ", " may", " june ", " jun ", " july " ," jul "]
#     Months2 = [" august " ," aug " ," septempber " ," sep ", " october " ," oct ", " november " ," nov " ," december ", " dec "]
    
    
#     rules = [
#         phone_pattern_1,phone_pattern_2,phone_pattern_3,
#         fax_pattern_1,fax_pattern_2,
#         website_pattern,
#         date_pattern1,date_pattern12,date_pattern13,date_pattern2,date_pattern22,date_pattern23,
#         date_pattern1y,date_pattern12y,date_pattern13y,date_pattern2y,date_pattern22y,date_pattern23y,
#         patient_id]
#     rules.extend(DaysOfWeeks)
#     rules.extend(Months)
#     rules.extend(Months2)
    
#     location = [0]*len(text)
    
#     for rule in rules:
#         for match in re.finditer(rule, text):
#             i, j = match.span()
#             location[i:j] = [1]*(j-i)
#     location2 = [location[i] if text[i] != ' ' else 0 for i in range(len(text))]

#     annotations = pd.DataFrame(columns=[
#         'document_id', 'annotation_id', 'start', 'stop', 'entity', 'entity_type', 'comment' 
#     ])
#     is_start = False 
#     for i in range(len(location2)):
#         if location2[i] == 1 and not is_start:
#             start = i 
#             is_start = True
#         if location2[i] == 0 and is_start:
#             assert (start is not None)
#             stop = i 
#             is_start = False  
#             entity = text[start:stop]
#             entity_type = 'PHI'
#             annotations = annotations.append({
#                 'document_id': document_id, 'annotation_id':'', 'start': start, 'stop':stop,
#                 'entity': entity, 'entity_type': entity_type, 'comment':''
#             }, ignore_index=True)
#             start, stop = None, None

#     return annotations
            

def argparser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help=(
            "The input data dir. Should contain the training "
            "files for the NER task."
        ),
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )

    return parser

def main():
    parser = argparser()
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    if args.output_folder is not None:
        output_folder = Path (args.output_folder)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        output_header = [
            'document_id', 'annotation_id', 'start', 'stop', 'entity',
            'entity_type', 'comment'
        ]
    else:
        output_folder = None 

    files = os.listdir(data_path)
    files = [f for f in files if f.endswith('.txt')]



    for fn in tqdm(files, total=len(files)):
        # load text file
        with open(data_path/f'{fn}', 'r') as fp:
            text = ''.join(fp.readlines())

        document_id = fn[:-len('.txt')]
        ex_preds = rule_based(document_id, text)
        if output_folder is not None:
            ex_preds.to_csv(output_folder/f'{document_id}.pred', index=False)




if __name__ == '__main__':
    main()
            


