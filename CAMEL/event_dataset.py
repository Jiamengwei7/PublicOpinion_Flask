import json
import os
import random
import copy
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import sys
from PIL import Image
from PIL import ImageFile
import os, glob
from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
# from transform.randaugment import RandomAugment

import torch
import pickle
import random

from tqdm import tqdm
import json
import pickle
import os

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths
# 为imSitu动词创建一个映射字典，并提供一些针对特定动词的映射和处理。
LABELS_list= ['None','Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Arrest-Jail', 'Life:Die', 'Movement:Transport', 'Transaction:Transfer-Money']

class ACE_Dataset(Dataset):
    def __init__(self, dataset, seq_len = 180):
        self.data = dataset
        self.data_length = len(dataset)
        self.seq_len = seq_len

    def __len__(self):
        return self.data_length
# __getitem__()方法是在对对象使用索引操作符[]时被调用
    def __getitem__(self, index):
        # data_item = self.data[index]
        data_item = self.data
        print(data_item[0])
        sub_word_lens = [len(data_item[0])]
        max_sub_word_len = self.seq_len

        original_sentence_len = [len(data_item[2])]
        max_original_sentence_len = self.seq_len

        data_x = data_item[0]
        data_span = data_item[1]
        text = ' '.join(data_item[-1])
        print("data_item[-1]",data_item[-1])
        words = data_item[-4]
        print("data_item[-2]",data_item[-2])
        images = data_item[-3]
        # print("imags:",images)
        # print("type(images)", type(images))
        f = torch.FloatTensor
        l = torch.LongTensor

        data_x = pad_sequence_to_length(data_x, max_sub_word_len)   # 将data_x填充到最大长度180
        bert_mask = get_mask_from_sequence_lengths(f(sub_word_lens), max_sub_word_len)

        # default_y = -1
        # data_y = pad_sequence_to_length(data_y, max_original_sentence_len, default_value=lambda: default_y)

        sequence_mask = get_mask_from_sequence_lengths(f(original_sentence_len), max_original_sentence_len)

        data_span_tensor = np.zeros((max_original_sentence_len, 2), dtype=int)

        temp = data_span[:max_original_sentence_len]

        for elem in temp:
            if elem[0] >= self.seq_len:
                elem[0] = self.seq_len - 1
            if elem[1] >= self.seq_len:
                elem[1] = self.seq_len - 1

        data_span_tensor[:len(temp), :] = temp

        dict = {'data_x':l(data_x),
                'bert_mask':bert_mask,
                'data_span_tensor':l(np.array(data_span_tensor)),
                'sequence_mask':sequence_mask,
                'images':images,
                'text':text,
                'words':words,
                }

        return dict