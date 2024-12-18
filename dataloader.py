from __future__ import print_function

import json

import h5py
from PIL import Image as pil_image
from torch import nn
from torchvision.transforms import InterpolationMode
from data_utils import img_feat_path_load, proc_img_feat
import random
import os
import re
import numpy as np
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchnet as tnt
import en_vectors_web_lg
import warnings
import glob
warnings.filterwarnings('ignore')

class FSL_VQA(data.Dataset):
    def __init__(self, root, partition='train', category='COCO', max_token=15, token_to_ix=None):
        super(FSL_VQA, self).__init__()
        self.root = root
        self.partition = partition
        self.split = split
        self.data_size = [3, 224, 224]
        self.max_token = max_token
        self.iid_to_img_feat_path = {}

        IMAGE_PATH = os.path.join(self.root, category)
        if category == 'COCO':
            txt_path = os.path.join(self.root, 'object', '{}.pth'.format(self.partition))
        else:
            txt_path = os.path.join(self.root, '{}.pth'.format(self.partition))

        if category == 'VQA':
            self.img_feat_path_list = glob.glob(self.root + '/vinvl/' + '*.npz')

        else:
            self.img_feat_path_list = glob.glob('' + '*.npz')

        self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)   

        self.c = np.load('/predict.npy', allow_pickle=True).item()

        datas = torch.load(txt_path)
        self.que_id, self.que_emb = tokenize(datas['all_words'])
        all_cls = []
        for c_list in self.c.values():
            for word in c_list:
                all_cls.append(word)
        self.cls_id, self.cls_emb = tokenize_class(all_cls, self.que_id, self.que_emb)
       
    # set normalizer
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.Resize(224, interpolation = InterpolationMode.BICUBIC),
                                                 transforms.RandomCrop(224, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

            self.token_to_ix, self.pretrained_emb = self.cls_id, self.cls_emb
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([transforms.Resize(224, interpolation = InterpolationMode.BICUBIC),
                                                 transforms.CenterCrop(224),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
            self.token_to_ix = token_to_ix
        self.token_size = len(self.token_to_ix)
        print('Loading {} dataset -phase {}, word size {}'.format(category, partition, self.token_size))
        
        image_path = []
        que = []
        label = []
        vinvl_path = []
        for line in datas['data']:
            que.append(line['question'])
            image_path.append(os.path.join(IMAGE_PATH, line['img_path']))
            label.append(datas['answers_index'][line['answer']])
            if len(self.iid_to_img_feat_path) > 0:
                vinvl_path.append(self.iid_to_img_feat_path[str(line['img_id'])])

        self.image_path = image_path
        self.question = que
        self.labels = label
        self.vinvl_path = vinvl_path
        self.full_class_list = list(np.unique(np.array(label)))
        self.label2ind = buildLabelIndex(self.labels)


    def __getitem__(self, index):
        path, que, label = self.image_path[index], self.question[index], self.labels[index]
        image_data = pil_image.open(path).convert('RGB')
        que = proc_ques(que, self.token_to_ix, self.max_token)
        img_id = path.split('/')[-1].split('_')[-1].split('.')[0]
        img_id = int(img_id)
        if len(self.vinvl_path) > 0:
            vinvl_file = np.load(self.vinvl_path[index])
            vinvl_feat = vinvl_file['x'].transpose((1, 0))
            vinvl_feat = proc_img_feat(vinvl_feat, 50)

        cls = self.c[str(img_id)]

        cls_id = proc_class(cls, self.cls_id, 50)

        return image_data, que, label, vinvl_feat, cls_id

    def __len__(self):
        return len(self.image_path)
    

class DataLoader:
    def __init__(self, dataset, num_tasks, num_ways, num_shots, num_queries, epoch_size, num_workers=8, batch_size=1):

        self.dataset = dataset
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.data_size = dataset.data_size
        self.full_class_list = dataset.full_class_list
        self.label2ind = dataset.label2ind
        self.transform = dataset.transform
        self.phase = dataset.partition
        self.root = dataset.root
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

    def get_task_batch(self):
        # init task batch data
        support_data, support_que, support_label, query_data, query_que, query_label, support_vinvl,\
            query_vinvl, support_cls, query_cls= [], [], [], [], [], [], [], [], [], []

        for _ in range(self.num_ways * self.num_shots):
            data = np.zeros(shape=[self.num_tasks] + self.data_size,
                            dtype='float32')
            ques = np.zeros(shape=[self.num_tasks, self.dataset.max_token],
                             dtype='int32')
            label = np.zeros(shape=[self.num_tasks],
                             dtype='float32')
            vinvl = np.zeros(shape=[self.num_tasks] + [50, 2048],
                             dtype='float32')
            cls = np.zeros(shape=[self.num_tasks, 50],
                             dtype='int32')
            support_data.append(data)
            support_que.append(ques)
            support_label.append(label)
            support_vinvl.append(vinvl)
            support_cls.append(cls)

        for _ in range(self.num_ways * self.num_queries):
            data = np.zeros(shape=[self.num_tasks] + self.data_size,
                            dtype='float32')
            ques = np.zeros(shape=[self.num_tasks, self.dataset.max_token],
                             dtype='int32')
            label = np.zeros(shape=[self.num_tasks],
                             dtype='float32')
            vinvl = np.zeros(shape=[self.num_tasks] + [50, 2048],
                             dtype='float32')
            cls = np.zeros(shape=[self.num_tasks, 50],
                             dtype='int32')
            query_data.append(data)
            query_que.append(ques)
            query_label.append(label)
            query_vinvl.append(vinvl)
            query_cls.append(cls)

        # for each task
        for t_idx in range(self.num_tasks):
            task_class_list = random.sample(self.full_class_list, self.num_ways)

            # for each sampled class in task
            for c_idx in range(self.num_ways):
                data_idx = random.sample(self.label2ind[task_class_list[c_idx]], self.num_shots + self.num_queries)
                class_data_list = [self.dataset[img_idx][0] for img_idx in data_idx]        
                class_ques_list = [self.dataset[img_idx][1] for img_idx in data_idx]
                vinvl_list = [self.dataset[img_idx][3] for img_idx in data_idx]
                cls_list = [self.dataset[img_idx][4] for img_idx in data_idx]
                for i_idx in range(self.num_shots):
                    # set data
                    support_data[i_idx + c_idx * self.num_shots][t_idx] = self.transform(class_data_list[i_idx])   
                    support_que[i_idx + c_idx * self.num_shots][t_idx] = class_ques_list[i_idx]
                    support_label[i_idx + c_idx * self.num_shots][t_idx] = c_idx
                    support_vinvl[i_idx + c_idx * self.num_shots][t_idx] = vinvl_list[i_idx]
                    support_cls[i_idx + c_idx * self.num_shots][t_idx] = cls_list[i_idx]

                # load sample for query set
                for i_idx in range(self.num_queries):
                    query_data[i_idx + c_idx * self.num_queries][t_idx] = \
                        self.transform(class_data_list[self.num_shots + i_idx])
                    query_que[i_idx + c_idx * self.num_queries][t_idx] = class_ques_list[self.num_shots + i_idx]
                    query_label[i_idx + c_idx * self.num_queries][t_idx] = c_idx
                    query_vinvl[i_idx + c_idx * self.num_queries][t_idx] = vinvl_list[self.num_shots + i_idx]
                    query_cls[i_idx + c_idx * self.num_queries][t_idx] = cls_list[self.num_shots + i_idx]

        support_data = torch.stack([torch.from_numpy(data).float() for data in support_data], 1)
        support_que = torch.stack([torch.from_numpy(data) for data in support_que], 1)
        support_label = torch.stack([torch.from_numpy(label).float() for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float() for data in query_data], 1)
        query_que = torch.stack([torch.from_numpy(data) for data in query_que], 1)
        query_label = torch.stack([torch.from_numpy(label).float() for label in query_label], 1)
        support_vinvl = torch.stack([torch.from_numpy(data).float() for data in support_vinvl], 1)
        query_vinvl = torch.stack([torch.from_numpy(data).float() for data in query_vinvl], 1)
        support_cls = torch.stack([torch.from_numpy(data).float() for data in support_cls], 1)
        query_cls = torch.stack([torch.from_numpy(data).float() for data in query_cls], 1)

        return support_data, support_que, support_label, query_data, query_que, query_label, support_vinvl, query_vinvl, support_cls, query_cls

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            support_data, support_que, support_label, query_data, query_que, query_label, \
            support_vinvl, query_vinvl,support_cls, query_cls = self.get_task_batch()
            return support_data, support_que, support_label, query_data, query_que, query_label, support_vinvl, query_vinvl, support_cls, query_cls

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(False if self.is_eval_mode else True))
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size


def data2datalabel(ori_data):
    data = []
    label = []
    for c_idx in ori_data:
        for i_idx in range(len(ori_data[c_idx])):
            data.append(ori_data[c_idx][i_idx])
            label.append(c_idx)
    return data, label


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


def split(que):
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        que.lower()
    ).replace('-', ' ').replace('/', ' ').split()
    return words


def tokenize(total_words, use_glove=True):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    pretrained_emb = []

    spacy_tool = None
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    for word in total_words:
        if word not in token_to_ix:
            token_to_ix[word] = len(token_to_ix)
            if use_glove:
                pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb


def tokenize_class(total_words, ix, emb, use_glove=True):
    token_to_ix = ix

    pretrained_emb = []

    spacy_tool = None
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()

    for word in total_words:
        word = word.replace(" ","")
        if word not in token_to_ix:
            token_to_ix[word] = len(token_to_ix)
            if use_glove:
                pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)
    pretrained_emb = np.concatenate((emb, pretrained_emb))

    return token_to_ix, pretrained_emb


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = split(ques)
    
    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def proc_class(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    for ix, word in enumerate(ques):
        word = word.replace(" ","")

        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


if __name__ == '__main__':

    dataset_train = FSL_VQA(root='.', partition='train')
    epoch_size = len(dataset_train)
    dloader_train = DataLoader(dataset_train,
                              num_tasks=25,  # 25
                              num_ways=5,     # 5
                              num_shots=1,   # 1
                              num_queries=1,  # 1
                              epoch_size=10000)
    bnumber = len(dloader_train)
    for epoch in range(0, 3):
        for idx, batch in enumerate(dloader_train(epoch)):
            print("epoch: ", epoch, "iter: ", idx)







