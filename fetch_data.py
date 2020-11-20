import pandas as pd
from transformers import BertTokenizer

data_dir = "./ner_dataset.csv"
data_set = pd.read_csv(data_dir ,encoding = "ISO-8859-1")


MODEL_PATH = "model.bin"

data_set.isnull().sum()
data_set = data_set.fillna(method='ffill')

'''loading  bert-base-uncased Tokenizer'''
TOKENIZER = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)
''' 

print("Data set info ")

list(data_set.columns)

data_set.head()

len(data_set.Word.unique()),len(data_set.POS.unique()),len(data_set.Tag.unique())

data_set['Sentence #'].nunique(), data_set.Word.nunique(), data_set.Tag.nunique()

data_set.groupby('Tag').size().reset_index(name='counts')
'''
