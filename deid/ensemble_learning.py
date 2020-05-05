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
from bert_deid import utils
import numpy as np
import re

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn import svm 
import pickle
import csv

import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from sklearn import metrics


logger = logging.getLogger()
logger.setLevel(logging.WARNING)

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
        "--text_path",
        default=None,
        type=str,
        required=True,
        help=(
            "The text dir. Should contain the training "
            "files for the NER task."
        ),
    )
    parser.add_argument(
        "--ref_path",
        default=None,
        type=str,
        required=True,
        help=(
            "The reference annotation dir. Should contain the training "
            "files for the NER task."
        ),
    )
    parser.add_argument(
        "--pred_dir",
        default=None,
        required=True,
        type=str,
        nargs='+',
        help= "The prediction dir."
    )
    parser.add_argument(
        "--output_pickle",
        default=None,
        type=str,
        required=True,
        help=
        "The output pickle where the best SVM model",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--do_train",
        action='store_true',
        help=
        "Train SVM classifier",
    )
    parser.add_argument(
        "--do_test",
        action='store_true',
        help=
        "Test SVM classifier",
    )
    

    return parser

def process_annotation(anno, document_id, text, is_expand=True):
    new_anno = pd.DataFrame(columns=anno.columns)
    text_pred = np.zeros(len(text), dtype=int)
    for i, row in anno.iterrows():
        text_pred[row['start']:row['stop']] = 1

    pattern = re.compile(r'\s')
    # find phi tokens split by whitespace
    for token, start, stop in utils.pattern_spans(text, pattern):
        if is_expand:
            # if any of individual chars are flagged,
            # whole token is flagged
            token_pred = np.any(text_pred[start:stop])
        else:
            token_pred = np.all(text_pred[start:stop])

        if token_pred:
            new_anno = new_anno.append({
                'document_id': document_id,
                'annotation_id':'',
                'start':start, 'stop': stop, 'entity': token, 'entity_type': 'PHI',
                'comment': ''
            }, ignore_index=True)

    return new_anno

def get_phi_loc_dict(all_model_annos):
    # get all distinct phi instance (with start, stop) predicted by all models
    phi_loc_dict = {}
    for i, row in all_model_annos.iterrows():
        start, stop = row['start'], row['stop']
        if (start, stop) not in phi_loc_dict:
            phi_loc_dict[(start, stop)] = len(phi_loc_dict)
    return phi_loc_dict

def exact_match(all_model_annos, phi_loc_dict, num_models):
    # num_phi_loc x num_model (feature)
    Xs = np.zeros((len(phi_loc_dict), num_models), dtype=int)
    for (start, stop) in phi_loc_dict.keys():
        match_rows = all_model_annos[(all_model_annos['start']==start) & (all_model_annos['stop']==stop)]
        match_rows = match_rows['model'].tolist()
        Xs[phi_loc_dict[(start, stop)], match_rows] = 1
    return Xs

def partial_match(all_model_annos, phi_loc_dict, num_models, len_text):
    pred_tars = np.zeros((num_models, len_text), dtype=int)
    for i in range(num_models):
        model_anno = all_model_annos[all_model_annos['model'] == i]
        for j, row in model_anno.iterrows():
            start, stop = row['start'], row['stop']
            pred_tars[i, start:stop] = 1

    # num_phi x num_model
    Xs = np.zeros((len(phi_loc_dict), num_models), dtype=int)

    for (start, stop) in phi_loc_dict.keys():
        # find partial match in each model
        model_partial_matched = []
        for j in range(num_models):
            # if any character in the interval is flagged
            # phi partially match by model j
            for k in range(start, stop):
                if pred_tars[j, k]:
                    model_partial_matched.append(j)
                    break
        Xs[phi_loc_dict[(start, stop)], model_partial_matched] = 1

    return Xs

def contain_conjunction_preposition(phi_loc_dict, text):
    # one word prepositions from
    # https://www.english-grammar-revolution.com/list-of-prepositions.html
    f = open('prepositions.txt', 'r')
    prepositions = set([line.split()[0] for line in f])
    f.close()
    # one word conjunctions from
    # https://www.english-grammar-revolution.com/list-of-conjunctions.html
    f = open('conjunctions.txt', 'r')
    conjunctions = set([line.split()[0] for line in f])
    f.close()

    # num_phi x num_model (feature)
    Xs = np.zeros((len(phi_loc_dict), 1), dtype=int)
    for (start, stop) in phi_loc_dict.keys():
        entity = text[start:stop].lower()
        if entity in prepositions or entity in conjunctions:
            Xs[phi_loc_dict[(start, stop)]] = 1
    return Xs

def count_span_phi(exact_match_Xs):
    # How many times the span of a PHI instance was predicted?
    # essentially count number of models predict exact start and stop loc
    # num_phi x num_models
    Xs = np.sum(exact_match_Xs, axis=1).reshape(-1, 1)
    return Xs

def count_phi(all_model_annos, phi_loc_dict, text):
    # How many times a PHI instance was predicted in a same clinical record?
    # does not care about location
    phi_freq = {}
    for i, row in all_model_annos.iterrows():
        entity = row['entity']
        phi_freq[entity] = phi_freq.get(entity, 0) + 1
    Xs = np.zeros((len(phi_loc_dict), 1), dtype=int)
    for (start, stop) in phi_loc_dict.keys():
        entity = text[start:stop]
        Xs[phi_loc_dict[(start, stop)]] = phi_freq[entity]
    return Xs

def combine_features(all_model_annos, phi_loc_dict, num_models, text):

    ex_match_Xs = exact_match(all_model_annos, phi_loc_dict, num_models)
    part_match_Xs = partial_match(all_model_annos, phi_loc_dict, num_models, len(text))
    contained_Xs = contain_conjunction_preposition(phi_loc_dict, text)
    span_phi_freq_Xs = count_span_phi(ex_match_Xs)
    phi_freq_Xs = count_phi(all_model_annos, phi_loc_dict, text)
    # print ('ex_match', ex_match_Xs.shape)
    # print ('part', part_match_Xs.shape)
    # print ('conta', contained_Xs.shape)
    # print ('span', span_phi_freq_Xs.shape)
    # print ('freq', phi_freq_Xs.shape)
    Xs = np.concatenate([
        ex_match_Xs,
        part_match_Xs,
        contained_Xs,
        span_phi_freq_Xs,
        phi_freq_Xs,
    ], axis=1)
    return Xs 

# def update_overall_phi_freq(all_model_annos, overall_freq):
#     for i, row in all_model_annos.iterrows():
#         entity = row['entity']
#         overall_freq[entity] = overall_freq.get(entity, 0) + 1

def find_true_phi_loc(true_anno, len_text):
    loc = np.zeros(len_text, dtype=int)
    for i, row in true_anno.iterrows():
        loc[row['start']:row['stop']] = 1
    return loc

def get_Ys(phi_loc_dict, true_phi_locs):
    Ys = np.zeros(len(phi_loc_dict), dtype=int)
    for (start, stop) in phi_loc_dict.keys():
        if np.all(true_phi_locs[start:stop]) == 1:
            Ys[phi_loc_dict[(start, stop)]] = 1
    return Ys

def get_info(phi_loc_dict, text):
    info = []
    for (start, stop) in phi_loc_dict.keys():
        info.append((start, stop, text[start:stop]))
    return info


def get_XYs(pred_dirs, data_path, ref_path):
    files = os.listdir(data_path / 'txt')
    files = [f for f in files if f.endswith('.txt')]

    #overall_freq = {}
    all_Xs = None
    all_Ys = None
    file_start = None
    file_length = None
    file_id = None
    phi_locs = None
    for fn in tqdm(files, total=len(files)):
        document_id = fn[:-len('.txt')]
        # load text file
        with open(data_path / 'txt' / fn, 'r') as fp:
            text = ''.join(fp.readlines())

        all_model_annos = pd.DataFrame(columns=[
            'model','document_id', 'annotation_id', 'start', 'stop', 'entity',
            'entity_type', 'comment'
        ])
        for i, model_pred in enumerate(pred_dirs):
            model_fn_pred = model_pred / f'{document_id}.pred'
            model_fn_anno = pd.read_csv(
                model_fn_pred, header=0,delimiter=",",
                dtype={
                'entity':str, 'entity_type':str
            })

            # process prediction made by different models
            # to have consistent format
            model_fn_anno = process_annotation(model_fn_anno, document_id, text)
            model_fn_anno['model'] = [i] * len(model_fn_anno) 
            all_model_annos = all_model_annos.append(model_fn_anno, ignore_index=True)
        
        phi_loc_dict = get_phi_loc_dict(all_model_annos)
        Xs = combine_features(all_model_annos, phi_loc_dict, len(pred_dirs), text)
        #update_overall_phi_freq(all_model_annos, overall_freq)
        # load ground truth
        gs_fn = ref_path / f'{document_id}.gs'
        gs = pd.read_csv(
            gs_fn, header=0, delimiter=",",
            dtype={
                'entity':str,'entity_type':str
            }
        )
        true_phi_locs = find_true_phi_loc(gs, len(text))
        Ys = get_Ys(phi_loc_dict, true_phi_locs)

        if all_Xs is None:
            all_Xs = Xs
            all_Ys = Ys
            file_start = [0]
            file_length = [len(Ys)]
            file_id = [document_id]
            phi_locs = [get_info(phi_loc_dict, text)]
        else:
            file_start.append(len(all_Ys))
            file_length.append(len(Ys))
            file_id.append(document_id)
            phi_locs.append(get_info(phi_loc_dict, text))
            all_Xs = np.concatenate([all_Xs, Xs], axis=0)
            all_Ys = np.concatenate([all_Ys, Ys], axis=0)

    return all_Xs, all_Ys, file_start, file_length, file_id, phi_locs

def view_data(X, Y):
    pca = TSNE(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X)
    # print ('Explained variation per principal component: {}'.format(pca.explained_variance_ratio))
    print ('pca', pca_result.shape)

    new_cord = np.vstack((pca_result.T, Y)).T
    print ('new_cord', new_cord.shape)
    df = pd.DataFrame(data=new_cord, columns=("PCA-One", "PCA-Two", "Label"))
    print ('dataframe', df.head())
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="PCA-One", y="PCA-Two",
        hue="Label",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.3,
    )
    plt.savefig('tsne_plot.jpg')
    plt.show()

def get_df_dict(anno_path, data_path, ext):
    files = os.listdir(data_path / 'txt')
    files = [f for f in files if f.endswith('.txt')]

    all_df = {}
    for fn in tqdm(files, total=len(files)):
        document_id = fn[:-len('.txt')]
        # load text file
        with open(data_path / 'txt' / fn, 'r') as fp:
            text = ''.join(fp.readlines())

        df = anno_path / f'{document_id}.{ext}'
        df = pd.read_csv(
            df, header=0,delimiter=",",
            dtype={
            'entity':str, 'entity_type':str
        })

        df = process_annotation(df, document_id, text)
        all_df[document_id] = df
    
    return all_df

def compare(pred_dirs, data_path, ref_path):
    files = os.listdir(data_path / 'txt')
    files = [f[:-len('.txt')] for f in files if f.endswith('.txt')]
    all_gs = get_df_dict(ref_path, data_path, 'gs')
    tps = {}
    num_tps = {}
    for pred_path in pred_dirs:
        df = get_df_dict(pred_path, data_path, 'pred')
        tps[pred_path] = []
        num_tps[pred_path] = 0
        for document_id in files:
            gs_fn = all_gs[document_id]
            gs_fn = gs_fn[gs_fn['document_id'] == document_id]
            df_fn = df[document_id]
            df_fn = df_fn[df_fn['document_id'] == document_id]
            phi = set([(document_id, start, stop) for start, stop in zip(gs_fn['start'].values, gs_fn['stop'].values)])
            pred = set([(document_id, start, stop) for start, stop in zip(df_fn['start'].values, df_fn['stop'].values)])
            true_postives = phi.intersection(pred)
            num_tp = len(true_postives)
            tps[pred_path].append(true_postives)
            num_tps[pred_path] += num_tp
    return tps, num_tps

def true_phi(pred_dirs, data_path, ref_path):
    files = os.listdir(data_path / 'txt')
    files = [f[:-len('.txt')] for f in files if f.endswith('.txt')]
    all_gs = get_df_dict(ref_path, data_path, 'gs')
    true = []
    for document_id in files:
        gs_fn = all_gs[document_id]
        gs_fn = gs_fn[gs_fn['document_id'] == document_id]
        phi = set([(document_id, start, stop) for start, stop in zip(gs_fn['start'].values, gs_fn['stop'].values)])
        true.extend(phi)
    return true

def plot_venn(pred_dirs, data_path, ref_path):
    tps, num_tps = compare(pred_dirs, data_path, ref_path)
    true = true_phi(pred_dirs, data_path, ref_path)
    # pickle.dump(tps, open('tps.p', 'wb'))
    # pickle.dump(num_tps, open('num_tps', 'wb'))
    pickle.dump(true, open('true.p', 'wb'))


def main():
    parser = argparser()
    args = parser.parse_args()

    pred_dirs = [Path(pred_dir) for pred_dir in args.pred_dir]
    data_path = Path(args.data_dir)
    ref_path = Path(args.ref_path)
    # plot_venn(pred_dirs, data_path, ref_path)
    Xs, Ys, file_start, file_length, file_id, phi_locs = get_XYs(pred_dirs, data_path, ref_path)
    # view_data(Xs, Ys)
    all_pred_Ys = []

    if args.do_train and not args.do_test:

        # hyper-params to tune
        svc = svm.SVC(random_state=42, kernel='linear', class_weight='balanced')
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
        Cs = [0.001, 0.01, 0.1, 1, 10, 100]
        gammas = [0.001, 0.01, 0.1, 1, 10, 100, 'auto', 'scale']

        logger.info("*********** Hyper-parameter Search*********")
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svc, param_grid, cv=k_fold, scoring='f1', verbose=2)
        grid_search.fit(Xs, Ys)
        best_params = grid_search.best_params_
        logger.info('*********best params: {} *********'.format(best_params))
        logger.info('*********best F1 score: {} ********'.format(grid_search.best_score_))
        best_estimator = grid_search.best_estimator_

        pickle.dump(best_estimator, open(args.output_pickle, 'wb'))
    elif args.do_test and not args.do_train:
        best_estimator = pickle.load(open(args.output_pickle, 'rb'))
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
        for i, file_s in enumerate(file_start):
            file_e = file_length[i]
            document_id = file_id[i]
            info = phi_locs[i]
            file_Xs = Xs[file_s:file_s+file_e]
            file_Ys = Ys[file_s:file_s+file_e]
            pred_Ys = best_estimator.predict(file_Xs)
            all_pred_Ys.extend(pred_Ys)
            
            with open(output_folder / f'{document_id}.pred', 'w') as fp:
                csvwriter = csv.writer(fp)
                # header
                csvwriter.writerow(output_header)
                for start, stop, entity in info:
                    row = [
                        document_id, '', start, stop, entity, 'PHI', ''
                    ]
                    csvwriter.writerow(row)

        print ('accuracy', metrics.accuracy_score(Ys, all_pred_Ys))
        print ('classification report', metrics.classification_report(Ys, all_pred_Ys))

if __name__ == "__main__": 
    main()





    

    










