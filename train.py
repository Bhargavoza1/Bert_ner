import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from model import CustomModel
import fetch_data
import preprocess_data as p_data

batch_size = 8
epochs = 10

def train_fn(data_loader, model, optimizer, device, scheduler):
    '''train function. in this function we will do backpropagation.
       i.e.  backpropagation(derivation from output to input) will adjust weight and bias to minimize the error in our learning stage
    '''
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    '''eval function. in this function we dont need to do backpropagation. '''
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, _, loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)


if __name__ == "__main__":
    '''preprocessing the data'''
    train_dataset = p_data.EntityDataset(sentences=p_data.train_sentences, pos=p_data.train_pos, tags=p_data.train_tag)
    valid_dataset = p_data.EntityDataset(sentences=p_data.test_sentences, pos=p_data.test_pos, tags=p_data.test_tag)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=2
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=2
    )

    '''ini the model'''
    device = torch.device("cuda")
    model = CustomModel(num_tag=p_data.num_tag, num_pos=p_data.num_pos)
    '''allocating tensors inside gpu'''
    model.to(device)

    '''ini the optimizer'''
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #a = [ n for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    #b = [n for n, p in param_optimizer if  any(nd in n for nd in no_decay)]

    '''applying regularization to specific parameters only'''
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    len(p_data.train_sentences)
    num_train_steps = int(len(p_data.train_sentences) / batch_size * epochs)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(epochs):
        print(f"current epoch = {epoch}")
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), fetch_data.MODEL_PATH)
            best_loss = test_loss
