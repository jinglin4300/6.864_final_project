import csv
import os
import pickle
from pathlib import Path
from bisect import bisect_left, bisect_right
import logging

import numpy as np
from transformers import BertTokenizer
from bert_deid.model import Transformer
from bert_deid.processors import DeidProcessor
from bert_deid.label import LabelCollection, LABEL_SET, LABEL_MEMBERSHIP
from tqdm import tqdm
from bert_deid.run_stanford import run_stanfordNER

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='/enc_data/deid-gs/i2b2_2014/test',
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--output",
        default='preds.pkl',
        type=str,
        help="Output file",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Output folder for CSV stand-off annotations.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    if args.output_folder is not None:
        output_folder = Path(args.output_folder)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        output_header = [
            'document_id', 'annotation_id', 'start', 'stop', 'entity',
            'entity_type', 'comment'
        ]
    else:
        output_folder = None

    files = os.listdir(data_path / 'txt')
    files = [f for f in files if f.endswith('.txt')]
    data = []

    preds = None
    lengths = []
    offsets = []
    labels = []

    for f in tqdm(files, total=len(files), desc='Files'):
        with open(data_path / 'txt' / f, 'r') as fp:
            text = ''.join(fp.readlines())

        ex_lengths, ex_offsets = run_stanfordNER(text)

        if output_folder is not None:
            # output the data to this folder as .pred files
            with open(output_folder / f'{f[:-4]}.pred', 'w') as fp:
                csvwriter = csv.writer(fp)
                # header
                csvwriter.writerow(output_header)
                for i in range(len(ex_offsets)):
                    start, stop = ex_offsets[i], ex_offsets[i] + ex_lengths[i]
                    entity = text[start:stop]
                    entity_type = 'PHI'
                    row = [
                        f[:-4],
                        str(i + 1), start, stop, entity, entity_type, None
                    ]
                    csvwriter.writerow(row)
        lengths.append(ex_lengths)
        offsets.append(ex_offsets)