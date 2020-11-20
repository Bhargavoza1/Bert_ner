import torch
from transformers import  BertModel
import torch.nn as nn

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)

    '''
    #my way of debugging tensor shape :P
    print()
    print(output.size())
    print(target.size())
    print(mask.size())
    print(num_labels)
    print(active_loss.size())
    print(active_logits.size())
    print(active_labels.size())
    print()
    '''

    return loss


class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag   #17
        self.num_pos = num_pos   #42
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop_1 = nn.Dropout(0.2)
        self.bert_drop_2 = nn.Dropout(0.2)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):

        # 8     ,      128         ,   x
        # batch , sentence length  , sth output

        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids) # output 8, 128, 768
        '''how we get o1 shape of 8,126,768? aaaaa it's real big process of BERT'''

        bo_tag = self.bert_drop_1(o1 ) # output 8, 128, 768
        bo_pos = self.bert_drop_2(o1 ) # output 8, 128, 768


        tag = self.out_tag(bo_tag) #output 8, 128, 17
        '''how we get tag shape of 8,126,17? by doind dot product of shape of '8, 128, 768' x '768, 17' '''
        pos = self.out_pos(bo_pos) #output 8, 128, 42
        '''how we get tag shape of 8,126,17? by doind dot product of shape of '8, 128, 768' x '768, 42' '''

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss