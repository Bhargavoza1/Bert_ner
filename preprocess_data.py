import pandas as pd
import numpy as np

import fetch_data as fd

from sklearn import preprocessing
from sklearn import model_selection

import joblib
import torch

'''  
_pos_encoder = preprocessing.LabelEncoder()
_tag_encoder = preprocessing.LabelEncoder()

fd.data_set.loc[:, "POS"] = _pos_encoder.fit_transform(fd.data_set["POS"])
fd.data_set.loc[:, "Tag"] = _tag_encoder.fit_transform(fd.data_set["Tag"])

sentences = fd.data_set.groupby("Sentence #")["Word"].apply(list).values
pos = fd.data_set.groupby("Sentence #")["POS"].apply(list).values
tag = fd.data_set.groupby("Sentence #")["Tag"].apply(list).values
'''


def process_data():
    _pos_encoder = preprocessing.LabelEncoder()
    _tag_encoder = preprocessing.LabelEncoder()

    '''# it will encode string to number 
        i.e. _pos_encoder.fit_transform(["paris", "paris", "tokyo", "amsterdam"])
        array([2, 2, 1]...)
    '''
    fd.data_set.loc[:, "POS"] = _pos_encoder.fit_transform(fd.data_set["POS"])
    fd.data_set.loc[:, "Tag"] = _tag_encoder.fit_transform(fd.data_set["Tag"])

    sentences = fd.data_set.groupby("Sentence #")["Word"].apply(list).values
    pos = fd.data_set.groupby("Sentence #")["POS"].apply(list).values
    tag = fd.data_set.groupby("Sentence #")["Tag"].apply(list).values
    return sentences,pos,tag





class EntityDataset:
        def __init__(self , data_set = process_data()):
                setence = data_set[0]
