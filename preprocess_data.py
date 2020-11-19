import pandas as pd
import numpy as np

import fetch_data as fd

from sklearn import preprocessing
from sklearn import model_selection

from transformers import BertTokenizer
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

sentences = fd.data_set.groupby("Sentence #")
sentences.groups
sentences

df = pd.DataFrame([('bird', 'Falconiformes', 389.0),
                 ('bird', 'Psittaciformes', 24.0),
                 ('mammal', 'Carnivora', 80.2),
                 ('mammal', 'Primates', 10),
                 ('mammal', 'Carnivora', 58)],
                index=['falcon', 'parrot', 'lion', 'monkey', 'leopard'],
                columns=('class', 'order', 'max_speed'))
 
grouped = df.groupby('max_speed')['class'].apply(list).to_numpy()
grouped

grouped2 = df.groupby('class')['max_speed'].apply(list).to_numpy()
grouped2

asd = preprocessing.LabelEncoder()

asd.fit(["paris", "paris", "tokyo", "amsterdam"])
asd.transform(["paris", "paris", "tokyo", "amsterdam"])

#or 

asd.fit_transform(["paris", "paris", "tokyo", "amsterdam"]
'''



def process_data():
    _pos_encoder = preprocessing.LabelEncoder()
    _tag_encoder = preprocessing.LabelEncoder()

    '''# it will encode string to number 
        i.e. _pos_encoder.fit_transform(["paris", "paris", "tokyo", "amsterdam"])
        array([1, 1, 2, 0])
    '''
    fd.data_set.loc[:, "POS"] = _pos_encoder.fit_transform(fd.data_set["POS"])
    fd.data_set.loc[:, "Tag"] = _tag_encoder.fit_transform(fd.data_set["Tag"])

    '''# groupby returns a Series with the data type of each column
    
         i.e. fd.data_set.groupby("Sentence #") will fetch every unique values from the column.
         
              fd.data_set.groupby("Sentence #")["Word"] will fetch values from Word column mapped with respect to every 
              unique values of Sentence #.
       # .apply(list) apply a function along an axis of the dataFrame.
       # .values return a numpy representation of the dataFrame.
         i.e DataFrame.to_numpy()
    '''
    sentences = fd.data_set.groupby("Sentence #")["Word"].apply(list).values
    pos = fd.data_set.groupby("Sentence #")["POS"].apply(list).values
    tag = fd.data_set.groupby("Sentence #")["Tag"].apply(list).values
    return sentences,pos,tag

sentences, pos, tag = process_data()

(train_sentences,test_sentences,
  train_pos,test_pos,
  train_tag,test_tag)  = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1)

sentences.shape
TOKENIZER =  BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)

class EntityDataset:
        def __init__(self ,texts, pos, tags ):
            self.texts = texts
            self.pos = pos
            self.tags = tags

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):

            text = self.texts[item]
            pos = self.pos[item]
            tags = self.tags[item]

            ids = []
            target_pos = []
            target_tag = []

            for i, s in enumerate(text):
                inputs = TOKENIZER.encode(
                    s,
                    add_special_tokens=False
                )
                # abhishek: ab ##hi ##sh ##ek
                input_len = len(inputs)
                ids.extend(inputs)
                target_pos.extend([pos[i]] * input_len)
                target_tag.extend([tags[i]] * input_len)

            ids = ids[:128 - 2]
            target_pos = target_pos[:128 - 2]
            target_tag = target_tag[:128 - 2]

            ids = [101] + ids + [102]
            target_pos = [0] + target_pos + [0]
            target_tag = [0] + target_tag + [0]

            mask = [1] * len(ids)
            token_type_ids = [0] * len(ids)

            padding_len = 128 - len(ids)

            ids = ids + ([0] * padding_len)
            mask = mask + ([0] * padding_len)
            token_type_ids = token_type_ids + ([0] * padding_len)
            target_pos = target_pos + ([0] * padding_len)
            target_tag = target_tag + ([0] * padding_len)

            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),

            }


device = torch.device("cuda")


a =EntityDataset(texts=train_sentences, pos=train_pos, tags=train_tag)

valid_data_loader = torch.utils.data.DataLoader(
    a, batch_size=8, num_workers=0
)

a.__len__()


from tqdm import tqdm




from transformers import    BertForPreTraining,BertTokenizer
model = BertForPreTraining.from_pretrained('bert-base-uncased', return_dict=True)
model.to(device)
len(valid_data_loader)
for data in tqdm(valid_data_loader, total=len(valid_data_loader)):
    for k, v in data.items():
        data[k] = v.to(device)
    _, loss = model(**data)








model = BertForPreTraining.from_pretrained('bert-base-uncased', return_dict=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits