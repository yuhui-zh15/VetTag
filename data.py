import os
import numpy as np
import torch
from torch.autograd import Variable
import logging
from collections import defaultdict
from os.path import join as pjoin
import json


def batchify(data, bsz):
    nbatch = data.shape[0] // bsz 
    data = data[:nbatch*bsz].reshape(bsz, nbatch)
    return data  


def pad_batch(batch, encoder, pad_start_end=True):
    # just build a numpy array that's padded
    max_len = np.max([len(x) for x in batch])
    if pad_start_end:
        max_len += 2
        for i in range(len(batch)):
            batch[i] = [encoder['_start_']] + batch[i] + [encoder['_end_']]
    max_len += 1
    padded_batch = np.full((len(batch), max_len), encoder['_pad_'])  # fill in pad_id
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            padded_batch[i, j] = batch[i][j]
    return padded_batch


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def np_to_var(np_obj, requires_grad=False):
    return Variable(torch.from_numpy(np_obj), requires_grad=requires_grad)


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, text, label, pad_id):
        self.text = np_to_var(text[:, :-1]).cuda()
        self.text_y = np_to_var(text[:, 1:]).cuda()
        self.text_lengths = (text[:, :-1] != pad_id).sum(axis=1) # <TODO> cuda or not 
        self.text_loss_mask = (self.text_y != pad_id).type(torch.float).cuda()
        self.text_mask = self.make_std_mask(self.text, pad_id)
        if len(label) != 0: self.label = np_to_var(label).cuda()

    @staticmethod
    def make_std_mask(tgt, pad_id):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad_id).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def get_data(encoder, data_dir, prefix, cut_down_len, label_size):
    # we are not padding anything in here, this is just repeating
    text = {}
    label = {}

    for data_type in ['train', 'valid', 'test']:
        text[data_type], label[data_type] = [], []
        data_path = pjoin(data_dir, prefix + "_" + data_type + ".tsv")
        if not os.path.exists(data_path): continue
        with open(data_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')
                tokens = [encoder.get(token, encoder['_unk_']) for token in columns[0].split()[:cut_down_len]]
                text[data_type].append(tokens)
                multi_label = np.zeros(label_size, dtype='float32') 
                if len(columns) == 2:
                    for number in map(int, columns[1].split()):
                        multi_label[number] = 1
                label[data_type].append(multi_label)
        assert len(text[data_type]) == len(label[data_type])
        text[data_type] = np.array(text[data_type])
        label[data_type] = np.array(label[data_type])
        logging.info('** {0} DATA : Found {1} pairs of {2} sentences.'.format(data_type.upper(), len(text[data_type]), data_type))

    train = {'text': text['train'], 'label': label['train']}
    valid = {'text': text['valid'], 'label': label['valid']}
    test = {'text': text['test'], 'label': label['test']}
    return train, valid, test
