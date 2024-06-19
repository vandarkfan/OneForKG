import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

def t5_er(filesave,dataset):
    checkpoint = torch.load(filesave + '/model_best_complex.tar')
    entity = checkpoint['model_state_dict']['embeddings.0.weight']
    relation = checkpoint['model_state_dict']['embeddings.1.weight']

    t5entity = torch.zeros([entity.shape[0], entity.shape[1]])
    t5relation = torch.zeros([relation.shape[0], relation.shape[1]])
    #准备map，准备两个不同数据集的id对应关系
    ssl_id2entity = dict()
    ssl_entity2id = dict()

    with open(filesave + '/ssl_ent_id', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    for line in lines:
        split = line.split('\t')
        ssl_entity2id[str(split[0])] = int(split[1])
        ssl_id2entity[int(split[1])] = str(split[0])

    t5_id2entity = dict()
    t5_entity2id = dict()

    with open('data/processed/'+dataset+'/entity2id.txt', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    for line in lines[1:]:
        split = line.split('\t')
        t5_entity2id[str(split[0])] = int(split[1])
        t5_id2entity[int(split[1])] = str(split[0])

    ssl_id2relation = dict()
    ssl_relation2id = dict()

    with open(filesave + '/ssl_rel_id', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    for line in lines:
        split = line.split('\t')
        ssl_relation2id[str(split[0])] = int(split[1])
        ssl_id2relation[int(split[1])] = str(split[0])

    t5_id2relation = dict()
    t5_relation2id = dict()

    with open('data/processed/'+dataset+'/relation2id.txt', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    for line in lines[1:]:
        split = line.split('\t')
        t5_relation2id[str(split[0])] = int(split[1])
        t5_id2relation[int(split[1])] = str(split[0])

    for i in tqdm(range(entity.shape[0])):
        t5entity[i] = entity[ssl_entity2id[t5_id2entity[i]]]


    for i in tqdm(range(relation.shape[0])):
        if i < relation.shape[0]/2:
            t5relation[i] = relation[ssl_relation2id[t5_id2relation[i]]]
        else:
            t5relation[i] = relation[ssl_relation2id[t5_id2relation[int(i - relation.shape[0]/2)]] + int(relation.shape[0]/2)]



    state = {'ent_embed': t5entity,
             'rel_embed': t5relation
             }
    torch.save(state, filesave + '/t5_complex_model.tar')


def t5_cluster(filesave,dataset,n_cluster):
    entity = torch.load(filesave + '/n_clusters' + str(n_cluster) +'.tar')
    t5entity = torch.zeros([entity.shape[0]])
    # 准备map，准备两个不同数据集的id对应关系
    ssl_id2entity = dict()
    ssl_entity2id = dict()

    with open(filesave +'/ssl_ent_id', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    for line in lines:
        split = line.split('\t')
        ssl_entity2id[str(split[0])] = int(split[1])
        ssl_id2entity[int(split[1])] = str(split[0])

    t5_id2entity = dict()
    t5_entity2id = dict()

    with open('data/processed/' + dataset + '/entity2id.txt', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    for line in lines[1:]:
        split = line.split('\t')
        t5_entity2id[str(split[0])] = int(split[1])
        t5_id2entity[int(split[1])] = str(split[0])

    for i in tqdm(range(entity.shape[0])):
        t5entity[i] = entity[ssl_entity2id[t5_id2entity[i]]]

    torch.save(t5entity, filesave + '/t5_cluster'+str(n_cluster)+'.tar')

if __name__ == '__main__':
    # t5_er('WN18RR')
    filesave = 'complex_wikipeople-1536'
    dataset = 'WikiPeople'
    # t5_er(filesave = filesave,dataset=dataset)
    t5_cluster(filesave = filesave,dataset=dataset,n_cluster = 30)