import os
import re
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.modified_model.modified_T5 import ModifiedT5ForConditionalGeneration
# from transformers import MT5ForConditionalGeneration
from transformers.optimization import Adafactor
from collections import Counter
from helper import get_performance
import json
import torch.nn.functional as F
import csv
class T5Finetuner(pl.LightningModule):
    def __init__(self, configs, ground_truth_dict, name_list_dict, prefix_trie_dict=None):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.pretrainKG = configs.pretrainKG
        if self.configs.dataset == 'WikiPeople':
            self.train_ground_truth = ground_truth_dict['train_ground_truth']
            self.all_ground_truth = ground_truth_dict['all_ground_truth']
        else:
            self.train_tail_ground_truth = ground_truth_dict['train_tail_ground_truth']
            self.train_head_ground_truth = ground_truth_dict['train_head_ground_truth']
            self.all_tail_ground_truth = ground_truth_dict['all_tail_ground_truth']
            self.all_head_ground_truth = ground_truth_dict['all_head_ground_truth']
        self.relname2id = name_list_dict['relname2id']
        self.entname2id = name_list_dict['entname2id']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.lmbda = 0.1
        self.prefix_trie = prefix_trie_dict['prefix_trie']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.next_token_dict = prefix_trie_dict['next_token_dict']
        if self.configs.tgt_descrip_max_length > 0:
            self.ent_token_ids_in_trie_with_descrip = prefix_trie_dict['ent_token_ids_in_trie_with_descrip']
        print(configs.gpu)
        self.T5ForConditionalGeneration = ModifiedT5ForConditionalGeneration.from_pretrained(configs.pretrained_model)
        if self.configs.dataset == 'NELL':
            self.typecons = json.load(open('data/processed/NELL/typecons.json'))
        if self.configs.dataset == 'wikizs':
            self.typecons = json.load(open('data/processed/wikizs/dev_candidates.json'))

        self.w = configs.w_SBeam
        if self.w >0.00001:
            checkpoint = torch.load('./' + self.configs.complex_dataset + '/t5_complex_model.tar')
            self.ent_embed = nn.Embedding.from_pretrained(checkpoint['ent_embed'])
            self.ent_embed.weight.requires_grad = False
            self.rel_embed = nn.Embedding.from_pretrained(checkpoint['rel_embed'])
            self.rel_embed.weight.requires_grad = False
            self.prompt_dim = checkpoint['rel_embed'].shape[-1]



        if self.configs.use_soft_prompt:
            prompt_dim = self.T5ForConditionalGeneration.model_dim
            self.rel_embed1 = nn.Embedding(self.configs.n_rel, prompt_dim)
            self.rel_embed2 = nn.Embedding(self.configs.n_rel, prompt_dim)
            if self.configs.use_rel_prompt_emb:
                self.rel_embed3 = nn.Embedding(self.configs.n_rel, prompt_dim)
                self.rel_embed4 = nn.Embedding(self.configs.n_rel, prompt_dim)
            else:
                self.ent_embed1 = nn.Embedding(self.configs.n_ent, prompt_dim)
                self.ent_embed2 = nn.Embedding(self.configs.n_ent, prompt_dim)

        self.history = {'perf': ..., 'loss': []}

    def training_step(self, batched_data, batch_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        if self.pretrainKG :
            src_ids = batched_data['source_ids']
            src_mask = batched_data['source_mask']
            target_ids = batched_data['target_ids']
            target_mask = batched_data['target_mask']
            # word_mask = batched_data['word_mask']
            input_entity_id = batched_data['input_entity_id']#[32,43]
            labels = target_ids.clone()
            labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100
            input_entity_id = torch.Tensor(input_entity_id)
            input_entity_id = torch.unsqueeze(input_entity_id, -1).long().to(src_ids.device)
            entity_hidden_state = self.ent_embed(input_entity_id)#[32,43,1536]
            entity_hidden_state = torch.reshape(entity_hidden_state, (entity_hidden_state.shape[0],2,-1))
            output = self.T5ForConditionalGeneration(input_ids=src_ids, attention_mask=src_mask, labels=labels,
                                                     entity_hidden_state=entity_hidden_state.to(src_ids.device))
            loss = torch.mean(output.loss)
            self.history['loss'].append(loss.detach().item())
        else:
            if self.configs.dataset != 'WikiPeople':
                loss_tail = self.training_step_dan(batched_data[0], batch_idx)
                loss_head = self.training_step_dan(batched_data[1], batch_idx)
                loss = (loss_tail + loss_head)/2
            else:
                loss_head = self.training_step_dan(batched_data, batch_idx)
                loss = loss_head

        return {'loss': loss}

    def training_step_dan(self, batched_data, batch_idx):
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # target_ids, target_mask, labels: .shape: (batch_size, padded_seq_len)
        target_ids = batched_data['target_ids']
        target_mask = batched_data['target_mask']
        sep = batched_data['sep']
        # sep_positon = batched_data['sep']
        labels = target_ids.clone()
        labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100
        # train_triples .shape: (batch_size, 3)
        train_triples = batched_data['train_triple']
        # ent_rel .shape: (batch_size, 2)
        # target_entity = torch.tensor(batched_data['target_ent']).to(src_ids.device)
        # entity_id_embed = self.ent_embed(ent_ids)
        addsource = None
        # if mode == 'head':
        #     rel_id_embed = self.rel_embed(rel_ids + self.configs.n_rel)  # 32,1536
        #     rel_id_embed_real, rel_id_embed_imag = rel_id_embed[:, :int(self.prompt_dim / 2)], rel_id_embed[:,
        #                                                                                        int(self.prompt_dim / 2):]
        #     ent_id_embed_real, ent_id_embed_imag = entity_id_embed[:, :int(self.prompt_dim / 2)], entity_id_embed[:,
        #                                                                                           int(self.prompt_dim / 2):]
        #     addsource = torch.zeros(
        #         [batched_data['source_ids'].shape[0], batched_data['source_ids'].shape[1], int(self.prompt_dim / 2)])
        #     for i in range(len(sep)):
        #         addsource[i, int(sep[i][0] +1):int((sep[i][0] + sep[i][1])/2), :] = rel_id_embed_real[i].unsqueeze(dim = 0).repeat(int((sep[i][0] + sep[i][1])/2) - int(sep[i][0] +1), 1)
        #         addsource[i, int((sep[i][0] + sep[i][1])/2):int(sep[i][1]), :] = rel_id_embed_imag[i].unsqueeze(dim = 0).repeat(int(sep[i][1]) - int((sep[i][0] + sep[i][1])/2), 1)
        #         addsource[i, int(sep[i][1] + 1):int((sep[i][1] + sep[i][2]) / 2), :] = ent_id_embed_real[i].unsqueeze(dim = 0).repeat(int((sep[i][1] + sep[i][2]) / 2) - int(sep[i][1] + 1), 1)
        #         addsource[i, int((sep[i][1] + sep[i][2]) / 2):int(sep[i][2]), :] = ent_id_embed_imag[i].unsqueeze(dim = 0).repeat(int(sep[i][2]) - int((sep[i][1] + sep[i][2]) / 2), 1)
        # else:
        #     rel_id_embed = self.rel_embed(rel_ids)
        #     rel_id_embed_real, rel_id_embed_imag = rel_id_embed[:, :int(self.prompt_dim / 2)], rel_id_embed[:,
        #                                                                                        int(self.prompt_dim / 2):]
        #     ent_id_embed_real, ent_id_embed_imag = entity_id_embed[:, :int(self.prompt_dim / 2)], entity_id_embed[:,
        #                                                                                           int(self.prompt_dim / 2):]
        #     addsource = torch.zeros(
        #         [batched_data['source_ids'].shape[0], batched_data['source_ids'].shape[1], int(self.prompt_dim / 2)])
        #     for i in range(len(sep)):
        #         addsource[i, :int(sep[i][0]/2), :] = ent_id_embed_real[i].unsqueeze(dim = 0).repeat(int(sep[i][0]/2), 1)
        #         addsource[i, int(sep[i][0]/2):int(sep[i][0]), :] = ent_id_embed_imag[i].unsqueeze(dim = 0).repeat(int(sep[i][0]) - int(sep[i][0]/2), 1)
        #         addsource[i, int(sep[i][0] + 1):int((sep[i][0] + sep[i][1]) / 2), :] = rel_id_embed_real[i].unsqueeze(dim = 0).repeat(int((sep[i][0] + sep[i][1]) / 2) - int(sep[i][0] + 1), 1)
        #         addsource[i, int((sep[i][0] + sep[i][1]) / 2):int(sep[i][1]), :] = rel_id_embed_imag[i].unsqueeze(dim = 0).repeat(int(sep[i][1]) - int((sep[i][0] + sep[i][1]) / 2), 1)

        # if self.configs.use_DT5:
        #     if mode == 'head':
        #         rel_id_embed = self.rel_embed(rel_ids + self.configs.n_rel)#32,1536
        #         rel_id_embed_real, rel_id_embed_imag = rel_id_embed[:, :int(self.prompt_dim/2)], rel_id_embed[:, int(self.prompt_dim/2):]
        #         ent_id_embed_real, ent_id_embed_imag = entity_id_embed[:, :int(self.prompt_dim/2)], entity_id_embed[:, int(self.prompt_dim/2):]
        #         addsource = torch.zeros([batched_data['source_ids'].shape[0], batched_data['source_ids'].shape[1], int(self.prompt_dim/2)])
        #         addsource[:, int(sep[0] +1):int((sep[0] + sep[1])/2), :] = rel_id_embed_real.unsqueeze(dim = 1).repeat(1, int((sep[0] + sep[1])/2) - int(sep[0] +1), 1)
        #         addsource[:, int((sep[0] + sep[1])/2):int(sep[1]), :] = rel_id_embed_imag.unsqueeze(dim = 1).repeat(1, int(sep[1]) - int((sep[0] + sep[1])/2), 1)
        #         addsource[:, int(sep[1] + 1):int((sep[1] + sep[2]) / 2), :] = ent_id_embed_real.unsqueeze(dim = 1).repeat(1, int((sep[1] + sep[2]) / 2) - int(sep[1] + 1), 1)
        #         addsource[:, int((sep[1] + sep[2]) / 2):int(sep[2]), :] = ent_id_embed_imag.unsqueeze(dim = 1).repeat(1, int(sep[2]) - int((sep[1] + sep[2]) / 2), 1)
        #         addsource.to(src_ids.device)
        #     else:
        #         rel_id_embed = self.rel_embed(rel_ids)
        #         rel_id_embed_real, rel_id_embed_imag = rel_id_embed[:, :int(self.prompt_dim/2)], rel_id_embed[:, int(self.prompt_dim/2):]
        #         ent_id_embed_real, ent_id_embed_imag = entity_id_embed[:, :int(self.prompt_dim/2)], entity_id_embed[:, int(self.prompt_dim/2):]
        #         addsource = torch.zeros([batched_data['source_ids'].shape[0], batched_data['source_ids'].shape[1], int(self.prompt_dim/2)])
        #         addsource[:, :int(sep[0]/2), :] = ent_id_embed_real.unsqueeze(dim = 1).repeat(1, int(sep[0]/2), 1)
        #         addsource[:, int(sep[0]/2):int(sep[0]), :] = ent_id_embed_imag.unsqueeze(dim = 1).repeat(1, int(sep[0]) - int(sep[0]/2), 1)
        #         addsource[:, int(sep[0] + 1):int((sep[0] + sep[1]) / 2), :] = rel_id_embed_real.unsqueeze(dim = 1).repeat(1, int((sep[0] + sep[1]) / 2) - int(sep[0] + 1), 1)
        #         addsource[:, int((sep[0] + sep[1]) / 2):int(sep[1]), :] = rel_id_embed_imag.unsqueeze(dim=1).repeat(1, int(
        #             sep[1]) - int((sep[0] + sep[1]) / 2), 1)
        #         addsource.to(src_ids.device)
        entity_hidden_state = None
        if self.configs.use_soft_prompt:
            # input_index .shape: (batch_size, seq_len + 4)
            input_index = batched_data['input_index']
            # soft_prompt_index .shape: (batch_size, 4)
            soft_prompt_index = batched_data['soft_prompt_index']
            inputs_emb, input_mask = self.get_soft_prompt_input_embed(src_ids, src_mask, ent_rel, input_index,
                                                                      soft_prompt_index)

        if self.configs.seq_dropout > 0.:
            if self.configs.use_soft_prompt:
                batch_size, length = input_mask.shape
                rand = torch.rand_like(input_mask.float())
                dropout = torch.logical_not(rand < self.configs.seq_dropout).long().type_as(input_mask)
                indicator_in_batch = torch.arange(batch_size).type_as(input_mask).unsqueeze(-1)

                trailing_mask = input_index[:, 1:] == 0
                input_index = (input_index + indicator_in_batch * src_ids.shape[1]).view(-1)
                input_src_ids = torch.index_select(src_ids.view(-1), 0, input_index).view(batch_size, length)
                input_src_ids[:, 1:][trailing_mask] = 0
                dropout[input_src_ids == 1820] = 1
                dropout[input_src_ids == 32099] = 1

                soft_prompt_index = (soft_prompt_index + indicator_in_batch * length).view(-1)
                dropout = dropout.view(-1)
                dropout[soft_prompt_index] = 1
                dropout = dropout.view(batch_size, length)
                input_mask = input_mask * dropout
            else:
                rand = torch.rand_like(src_mask.float())
                dropout = torch.logical_not(rand < self.configs.seq_dropout).long().type_as(src_mask)
                dropout[src_ids == 1820] = 1
                dropout[src_ids == 32099] = 1
                src_mask = src_mask * dropout

        if self.configs.use_soft_prompt:
            output = self.T5ForConditionalGeneration(inputs_embeds=inputs_emb, attention_mask=input_mask, labels=labels,
                                                     output_hidden_states=True)
        else:
            output = self.T5ForConditionalGeneration(input_ids=src_ids, attention_mask=src_mask, labels=labels,
                                                     entity_hidden_state=entity_hidden_state,addsource =addsource,sep = sep)
        loss = torch.mean(output.loss)

        # ent = output.encoder_last_hidden_state[:,:2,:].view(batched_data['source_ids'].shape[0],-1)
        # rel = output.encoder_last_hidden_state[:, 2:4, :].view(batched_data['source_ids'].shape[0],-1)

        # complex_predict = self.complex_s(mode=mode, ent_ids=ent_ids, rel_ids=rel_ids)
        # factors = self.get_factor(self.ent_embed(ent_ids), self.rel_embed(rel_ids), self.ent_embed(target_entity.to(0)))
        # l_reg = self.penalty(factors)
        # complex_f_loss = nn.CrossEntropyLoss(reduction='mean')
        # complex_loss = complex_f_loss(complex_predict.to(0), target_entity.to(0))
        # loss = (1 - self.w) * loss + self.w * (complex_loss + l_reg)
        self.history['loss'].append(loss.detach().item())
        return loss
    def penalty(self, factors):
        """

        :param factors: tuple, (s, p, o), batch_size * rank
        :return:
        """
        norm = 0
        for f in factors:
            norm += self.lmbda * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]
    def get_factor(self, lhs, rel, rhs):
        lhs = lhs[:, :int(self.prompt_dim/2)], lhs[:, int(self.prompt_dim/2):]
        rel = rel[:, :int(self.prompt_dim/2)], rel[:, int(self.prompt_dim/2):]
        rhs = rhs[:, :int(self.prompt_dim/2)], rhs[:, int(self.prompt_dim/2):]
        return (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))

    def complex_ds(self, mode, ent, rel):
        to_score_entity = self.ent_embed.weight
        to_score_entity = to_score_entity[:, :int(self.prompt_dim/2)], to_score_entity[:, int(self.prompt_dim/2):]
        if mode[0] == 'tail':
            lhs = ent
            rel = rel
            # rhs = self.ent_embed(target_entity[i])
            # rhs = torch.squeeze(rhs, dim=0)
            lhs = lhs[:,:int(self.prompt_dim/2)], lhs[:,int(self.prompt_dim/2):]
            rel = rel[:,:int(self.prompt_dim/2)], rel[:,int(self.prompt_dim/2):]
            # rhs = rhs[:self.prompt_dim], rhs[self.prompt_dim:]
            rhs_scores = (
                    (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_entity[0].transpose(0, 1) +
                    (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score_entity[1].transpose(0, 1)
            )
            scores = rhs_scores
        else:
            rhs = ent
            rel = rel
            rhs = rhs[:,:int(self.prompt_dim/2)], rhs[:,int(self.prompt_dim/2):]
            rel = rel[:,:int(self.prompt_dim/2)], rel[:,int(self.prompt_dim/2):]
            lhs_scores = (
                (rel[0] * rhs[0] + rel[1] * rhs[1]) @ to_score_entity[0].transpose(0, 1) +
                (rel[0] * rhs[1] - rel[1] * rhs[0]) @ to_score_entity[1].transpose(0, 1)
            )
            scores = lhs_scores
        return scores
    def complex_s(self, mode, ent_ids, rel_ids):
        to_score_entity = self.ent_embed.weight
        to_score_entity = to_score_entity[:, :int(self.prompt_dim/2)], to_score_entity[:, int(self.prompt_dim/2):]
        if mode[0] == 'tail':
            lhs = self.ent_embed(ent_ids)
            rel = self.rel_embed(rel_ids)
            # rhs = self.ent_embed(target_entity[i])
            # rhs = torch.squeeze(rhs, dim=0)
            lhs = lhs[:,:int(self.prompt_dim/2)], lhs[:,int(self.prompt_dim/2):]
            rel = rel[:,:int(self.prompt_dim/2)], rel[:,int(self.prompt_dim/2):]
            # rhs = rhs[:self.prompt_dim], rhs[self.prompt_dim:]
            rhs_scores = (
                    (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_entity[0].transpose(0, 1) +
                    (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score_entity[1].transpose(0, 1)
            )
            scores = rhs_scores
        else:
            rhs = self.ent_embed(ent_ids)
            rel = self.rel_embed(rel_ids + self.configs.n_rel)
            rhs = rhs[:,:int(self.prompt_dim/2)], rhs[:,int(self.prompt_dim/2):]
            rel = rel[:,:int(self.prompt_dim/2)], rel[:,int(self.prompt_dim/2):]
            lhs_scores = (
                (rel[0] * rhs[0] + rel[1] * rhs[1]) @ to_score_entity[0].transpose(0, 1) +
                (rel[0] * rhs[1] - rel[1] * rhs[0]) @ to_score_entity[1].transpose(0, 1)
            )
            scores = lhs_scores
        return scores
    def validation_step(self, batched_data, batch_idx, dataset_idx):
    # def validation_step(self, batched_data, batch_idx):
        if self.current_epoch < self.configs.skip_n_val_epoch:
            return
        if self.pretrainKG:
            src_ids = batched_data['source_ids']
            src_mask = batched_data['source_mask']
            target_ids = batched_data['target_ids']
            target_mask = batched_data['target_mask']
            input_entity_id = batched_data['input_entity_id']
            labels = target_ids.clone()
            labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100
            input_entity_id = torch.Tensor(input_entity_id)
            input_entity_id = torch.unsqueeze(input_entity_id, -1).long().to(src_ids.device)
            entity_hidden_state = self.ent_embed(input_entity_id)  # [32,43,1536]
            entity_hidden_state = torch.reshape(entity_hidden_state, (entity_hidden_state.shape[0], 2, -1))
            output = self.T5ForConditionalGeneration(input_ids=src_ids, attention_mask=src_mask, labels=labels,
                                                     entity_hidden_state=entity_hidden_state.to(src_ids.device))
            loss = torch.mean(output.loss)
            return {'loss': loss}
        else:
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len) .type: torch.tensor
            src_ids = batched_data['source_ids']
            src_mask = batched_data['source_mask']
            # src_names, target_names: .shape: (batch_size, ) .type:list(str)
            src_names = batched_data['source_names']
            target_names = batched_data['target_names']
            # test_triple: .shape: (batch_size, 3)
            self.test_triple = batched_data['test_triple']
            # ent_rel: .shape: (batch_size, 2)
            # target_entity = torch.tensor(batched_data['target_ent']).to(src_ids.device)
            mode = batched_data['mode']
            self.dataset_idx = dataset_idx

            sep = batched_data['sep']

            if self.configs.dataset != 'WikiPeople':
                self.ent_rel = batched_data['ent_rel']
                ent_ids, rel_ids = torch.squeeze(self.ent_rel[:, [0]]), torch.squeeze(self.ent_rel[:, [1]])


            # entity_id_embed = self.ent_embed(ent_ids)

            addsource = None
            # if mode == 'head':
            #     rel_id_embed = self.rel_embed(rel_ids + self.configs.n_rel)  # 32,1536
            #     rel_id_embed_real, rel_id_embed_imag = rel_id_embed[:, :int(self.prompt_dim / 2)], rel_id_embed[:,
            #                                                                                        int(self.prompt_dim / 2):]
            #     ent_id_embed_real, ent_id_embed_imag = entity_id_embed[:, :int(self.prompt_dim / 2)], entity_id_embed[:,
            #                                                                                           int(self.prompt_dim / 2):]
            #     addsource = torch.zeros(
            #         [batched_data['source_ids'].shape[0], batched_data['source_ids'].shape[1], int(self.prompt_dim / 2)])
            #     for i in range(len(sep)):
            #         addsource[i, int(sep[i][0] + 1):int((sep[i][0] + sep[i][1]) / 2), :] = rel_id_embed_real[i].unsqueeze(
            #             dim=0).repeat(int((sep[i][0] + sep[i][1]) / 2) - int(sep[i][0] + 1), 1)
            #         addsource[i, int((sep[i][0] + sep[i][1]) / 2):int(sep[i][1]), :] = rel_id_embed_imag[i].unsqueeze(
            #             dim=0).repeat(int(sep[i][1]) - int((sep[i][0] + sep[i][1]) / 2), 1)
            #         addsource[i, int(sep[i][1] + 1):int((sep[i][1] + sep[i][2]) / 2), :] = ent_id_embed_real[i].unsqueeze(
            #             dim=0).repeat(int((sep[i][1] + sep[i][2]) / 2) - int(sep[i][1] + 1), 1)
            #         addsource[i, int((sep[i][1] + sep[i][2]) / 2):int(sep[i][2]), :] = ent_id_embed_imag[i].unsqueeze(
            #             dim=0).repeat(int(sep[i][2]) - int((sep[i][1] + sep[i][2]) / 2), 1)
            # else:
            #     rel_id_embed = self.rel_embed(rel_ids)
            #     rel_id_embed_real, rel_id_embed_imag = rel_id_embed[:, :int(self.prompt_dim / 2)], rel_id_embed[:,
            #                                                                                        int(self.prompt_dim / 2):]
            #     ent_id_embed_real, ent_id_embed_imag = entity_id_embed[:, :int(self.prompt_dim / 2)], entity_id_embed[:,
            #                                                                                           int(self.prompt_dim / 2):]
            #     addsource = torch.zeros(
            #         [batched_data['source_ids'].shape[0], batched_data['source_ids'].shape[1], int(self.prompt_dim / 2)])
            #     for i in range(len(sep)):
            #         addsource[i, :int(sep[i][0] / 2), :] = ent_id_embed_real[i].unsqueeze(dim=0).repeat(int(sep[i][0] / 2), 1)
            #         addsource[i, int(sep[i][0] / 2):int(sep[i][0]), :] = ent_id_embed_imag[i].unsqueeze(dim=0).repeat(
            #             int(sep[i][0]) - int(sep[i][0] / 2), 1)
            #         addsource[i, int(sep[i][0] + 1):int((sep[i][0] + sep[i][1]) / 2), :] = rel_id_embed_real[i].unsqueeze(
            #             dim=0).repeat(int((sep[i][0] + sep[i][1]) / 2) - int(sep[i][0] + 1), 1)
            #         addsource[i, int((sep[i][0] + sep[i][1]) / 2):int(sep[i][1]), :] = rel_id_embed_imag[i].unsqueeze(
            #             dim=0).repeat(int(sep[i][1]) - int((sep[i][0] + sep[i][1]) / 2), 1)
        #以上为正版的区分器
            # if self.configs.use_DT5:
            #     if mode == 'head':
            #         rel_id_embed = self.rel_embed(rel_ids + self.configs.n_rel)#32,1536
            #         if len(rel_id_embed.shape) == 1:
            #             rel_id_embed = rel_id_embed.unsqueeze(0)
            #             entity_id_embed = entity_id_embed.unsqueeze(0)
            #         rel_id_embed_real, rel_id_embed_imag = rel_id_embed[:, :int(self.prompt_dim/2)], rel_id_embed[:, int(self.prompt_dim/2):]
            #         ent_id_embed_real, ent_id_embed_imag = entity_id_embed[:, :int(self.prompt_dim/2)], entity_id_embed[:, int(self.prompt_dim/2):]
            #         addsource = torch.zeros([batched_data['source_ids'].shape[0], batched_data['source_ids'].shape[1], int(self.prompt_dim/2)])
            #         addsource[:, int(sep[0] +1):int((sep[0] + sep[1])/2), :] = rel_id_embed_real.unsqueeze(dim = 1).repeat(1, int((sep[0] + sep[1])/2) - int(sep[0] +1), 1)
            #         addsource[:, int((sep[0] + sep[1])/2):int(sep[1]), :] = rel_id_embed_imag.unsqueeze(dim = 1).repeat(1, int(sep[1]) - int((sep[0] + sep[1])/2), 1)
            #         addsource[:, int(sep[1] + 1):int((sep[1] + sep[2]) / 2), :] = ent_id_embed_real.unsqueeze(dim = 1).repeat(1, int((sep[1] + sep[2]) / 2) - int(sep[1] + 1), 1)
            #         addsource[:, int((sep[1] + sep[2]) / 2):int(sep[2]), :] = ent_id_embed_imag.unsqueeze(dim = 1).repeat(1, int(sep[2]) - int((sep[1] + sep[2]) / 2), 1)
            #         addsource.to(src_ids.device)
            #     else:
            #         rel_id_embed = self.rel_embed(rel_ids)
            #         if len(rel_id_embed.shape) == 1:
            #             rel_id_embed = rel_id_embed.unsqueeze(0)
            #             entity_id_embed = entity_id_embed.unsqueeze(0)
            #         rel_id_embed_real, rel_id_embed_imag = rel_id_embed[:, :int(self.prompt_dim/2)], rel_id_embed[:, int(self.prompt_dim/2):]
            #         ent_id_embed_real, ent_id_embed_imag = entity_id_embed[:, :int(self.prompt_dim/2)], entity_id_embed[:, int(self.prompt_dim/2):]
            #         addsource = torch.zeros([batched_data['source_ids'].shape[0], batched_data['source_ids'].shape[1], int(self.prompt_dim/2)])
            #         addsource[:, :int(sep[0]/2), :] = ent_id_embed_real.unsqueeze(dim = 1).repeat(1, int(sep[0]/2), 1)
            #         addsource[:, int(sep[0]/2):int(sep[0]), :] = ent_id_embed_imag.unsqueeze(dim = 1).repeat(1, int(sep[0]) - int(sep[0]/2), 1)
            #         addsource[:, int(sep[0] + 1):int((sep[0] + sep[1]) / 2), :] = rel_id_embed_real.unsqueeze(dim = 1).repeat(1, int((sep[0] + sep[1]) / 2) - int(sep[0] + 1), 1)
            #         addsource[:, int((sep[0] + sep[1]) / 2):int(sep[1]), :] = rel_id_embed_imag.unsqueeze(dim = 1).repeat(1, int(sep[1]) - int((sep[0] + sep[1]) / 2), 1)
            #         addsource.to(src_ids.device)

            entity_hidden_state = None

            # if self.configs.dataset != 'WikiPeople':
            #     if dataset_idx == 0:
            #         self.all_ground_truth = self.all_tail_ground_truth
            #         self.train_ground_truth = self.train_tail_ground_truth
            #     else:
            #         self.all_ground_truth = self.all_head_ground_truth
            #         self.train_ground_truth = self.train_head_ground_truth

            if dataset_idx == 0:
                self.all_ground_truth = self.all_tail_ground_truth
                self.train_ground_truth = self.train_tail_ground_truth
            else:
                self.all_ground_truth = self.all_head_ground_truth
                self.train_ground_truth = self.train_head_ground_truth

            # generated_text .type: list(str) .len: batch_size * num_beams
            generated_text, scores_t5 = self.decode(src_ids, src_mask, batched_data, entity_hidden_state, addsource,sep)
            group_text = [generated_text[i:i + self.configs.num_beams] for i in range(0, len(generated_text), self.configs.num_beams)]
            # # #
            # # # #
            if self.w>0.00001:
                scores_t5 = scores_t5.contiguous().view(-1, self.configs.num_beams)
                scores_t5 = torch.softmax(scores_t5, dim=1)
                generated_id = -1 * torch.ones([len(generated_text)])
                for i in range(len(generated_text)):
                    if generated_text[i] in self.entname2id.keys():
                        generated_id[i] = self.entname2id[generated_text[i]]
                generated_id = generated_id.contiguous().view(-1, self.configs.num_beams)
                scores_complex = self.complex_s(mode = mode, ent_ids = ent_ids, rel_ids = rel_ids)
                scores_complex2t5 = -1 * torch.ones([generated_id.shape[0], generated_id.shape[1]])
                for i in range(generated_id.shape[0]):
                    for j in range(generated_id.shape[1]):
                        if generated_id[i,j]!=-1:
                            scores_complex2t5[i,j] = scores_complex[i, int(generated_id[i,j])]#[8,40]里面不对劲的玩意都变成-10000了，有数字的可能在-7~7之间
                        else:
                            scores_complex2t5[i, j] = -10000
                scores_complex2t5 = torch.softmax(scores_complex2t5, dim=1)
                scores = (1 - self.w) * scores_t5 + self.w * scores_complex2t5.to(src_ids.device)
                sorted_scores, indices = torch.sort(scores, descending=True, dim=-1)
                sorted_group_text = []
                for i in range(len(group_text)):
                    sorted_group_text_dan = []
                    for j in range(self.configs.num_beams):
                        sorted_group_text_dan.append(group_text[i][indices[i,j]])
                    sorted_group_text.append(sorted_group_text_dan)
                group_text = sorted_group_text


            # if self.configs.log_text:
            #     self.log_generation(group_text, src_names, target_names, batch_idx, dataset_idx)
            ranks = []


            for i, texts in enumerate(group_text):
                if self.configs.temporal:
                    hr_key = (self.test_triple[i][dataset_idx], self.test_triple[i][2], self.test_triple[i][3])
                else:
                    hr_key = (self.test_triple[i][dataset_idx], self.test_triple[i][2])

                all_gt_ids = self.all_ground_truth[hr_key]#list里面是这个实体应该过滤的id
                all_gt_seqs = [self.ent_name_list[ids] for ids in all_gt_ids]

                if mode[0] == 'tail':
                    if self.configs.dataset == 'NELL':
                        nell_ids = self.typecons[str(hr_key[1])]['tail']  # list里面是这个实体应该过滤的id
                        nell_fanwei = [self.ent_name_list[ids] for ids in nell_ids]
                    if self.configs.dataset == 'wikizs':
                        nell_ids = self.typecons[str(hr_key[0]) + '\t' + str(hr_key[1])]['tail']  # list里面是这个实体应该过滤的id
                        nell_fanwei = [self.ent_name_list[ids] for ids in nell_ids]
                ## get rank
                if target_names[i] in texts:
                    top_entities = set()
                    rank = 1
                    for text in texts:
                        if text == target_names[i]:
                            ranks.append(rank)
                            break
                        if text in set(self.ent_name_list) and (text not in all_gt_seqs) and (text not in top_entities) and self.configs.dataset != 'NELL':
                            top_entities.add(text)
                            rank += 1
                        if text in set(self.ent_name_list) and (text not in all_gt_seqs) and (text not in top_entities) and self.configs.dataset == 'NELL':
                            if mode[0]=='tail':
                                if text not in nell_fanwei:
                                    rank=rank
                                else:
                                    top_entities.add(text)
                                    rank += 1
                            else:
                                top_entities.add(text)
                                rank += 1
                    # if int(ent_ids[i])==20170 and int(rel_ids[i])==9 and mode[i]=='head':
                    #     print(int(ent_ids[i]))
                    #     print(self.ent_name_list[int(ent_ids[i])])
                    #     print('scores_t5:')
                    #     print(scores_t5[i, :])
                    #     print('scores_complex2t5:')
                    #     print(scores_complex2t5[i, :])
                    #     print('scores:')
                    #     print(scores[i, :])
                    #     print('sorted_scores:')
                    #     print(sorted_scores[i, :])
                    #     print('group_text:')
                    #     print(group_text[i])
                    # if int(ent_ids[i])==31015 and int(rel_ids[i])==4 and mode[i]=='head':
                    #     print(int(ent_ids[i]))
                    #     print(self.ent_name_list[int(ent_ids[i])])
                    #     print('scores_t5:')
                    #     print(scores_t5[i, :])
                    #     print('scores_complex2t5:')
                    #     print(scores_complex2t5[i, :])
                    #     print('scores:')
                    #     print(scores[i, :])
                    #     print('sorted_scores:')
                    #     print(sorted_scores[i, :])
                    #     print('group_text:')
                    #     print(group_text[i])
                    # if int(ent_ids[i])==8514 and int(rel_ids[i])==9 and mode[i]=='tail':
                    #     print(int(ent_ids[i]))
                    #     print(self.ent_name_list[int(ent_ids[i])])
                    #     print('scores_t5:')
                    #     print(scores_t5[i, :])
                    #     print('scores_complex2t5:')
                    #     print(scores_complex2t5[i, :])
                    #     print('scores:')
                    #     print(scores[i, :])
                    #     print('sorted_scores:')
                    #     print(sorted_scores[i, :])
                    #     print('group_text:')
                    #     print(group_text[i])
                    # with open("t5.csv", "a+") as csvfile:
                    #     writer = csv.writer(csvfile)
                    #     writer.writerow([int(ent_ids[i]), int(rel_ids[i]), mode[i], rank])

                else:
                    ranks.append(random.randint(self.configs.num_beams + 1, self.configs.n_ent))


            out = {'ranks': ranks}
            return out

    def decode(self, src_ids, src_mask, batched_data,entity_hidden_state, addsource,sep):
        def _extract(generated_text):
            if self.configs.tgt_descrip_max_length > 0:
                compiler = re.compile(r'<extra_id_0>(.*?)\[')
            else:
                compiler = re.compile(r'<extra_id_0>(.*)<extra_id_1>')
            extracted_text = []
            for text in generated_text:
                match = compiler.search(text)
                if match is None:
                    # text = text.strip().lstrip('<pad> <extra_id_0>')
                    extracted_text.append(text.strip())
                else:
                    extracted_text.append(match.group(1).strip())
            return extracted_text

        def _next_candidate(configs, batch_idx, input_ids):
            if self.configs.temporal:
                hr_key = (self.test_triple[batch_idx][self.dataset_idx], self.test_triple[batch_idx][2], self.test_triple[batch_idx][3])
            else:
                hr_key = (self.test_triple[batch_idx][self.dataset_idx], self.test_triple[batch_idx][2])

            all_gt_ids = self.all_ground_truth[hr_key]
            ent_token_ids_in_trie = self.ent_token_ids_in_trie_with_descrip if configs.tgt_descrip_max_length > 0 else self.ent_token_ids_in_trie
            all_gt_seq = [tuple(ent_token_ids_in_trie[ids]) for ids in all_gt_ids]

            pred_pos = 1 if self.dataset_idx == 0 else 0
            pred_ids = tuple(ent_token_ids_in_trie[self.test_triple[batch_idx][pred_pos]])

            input_ids = input_ids.tolist()
            if input_ids[0] == 0:
                input_ids = input_ids[1:]

            if tuple(input_ids) in self.next_token_dict:
                if len(input_ids) == 0:
                    return [32099]
                if input_ids[-1] == 32098:
                    return [1]
                next_tokens = self.next_token_dict[tuple(input_ids)]
                all_gt_seq = [seq for seq in all_gt_seq if tuple(seq[: len(input_ids)]) == tuple(input_ids)]
                gt_next_tokens = Counter([seq[len(input_ids)] for seq in all_gt_seq if len(input_ids) < len(seq)])
                if tuple(pred_ids[: len(input_ids)]) == tuple(input_ids) and len(input_ids) < len(pred_ids):
                    pred_id = Counter([pred_ids[len(input_ids)]])
                else:
                    pred_id = Counter([])
                next_tokens = list(set(next_tokens - gt_next_tokens + pred_id))
                return next_tokens
            else:
                return []

        if self.configs.decoder in ['beam_search', 'diverse_beam_search']:
            num_beam_groups = self.configs.num_beam_groups if self.configs.decoder == 'diverse_beam_search' else 1
            diversity_penalty = self.configs.diversity_penalty if self.configs.decoder == 'diverse_beam_search' else 0.
            prefix_allowed_tokens_fn = lambda batch_idx, input_ids: _next_candidate(self.configs, batch_idx, input_ids) if self.configs.use_prefix_search else None
            if self.configs.use_soft_prompt:
                # input_index .shape: (batch_size, seq_len + 4)
                input_index = batched_data['input_index']
                # soft_prompt_index .shape: (batch_size, 4)
                soft_prompt_index = batched_data['soft_prompt_index']
                inputs_emb, input_mask = self.get_soft_prompt_input_embed(src_ids, src_mask, self.ent_rel, input_index,
                                                                          soft_prompt_index)
                outputs = self.T5ForConditionalGeneration.generate(inputs_embeds=inputs_emb,
                                                                   attention_mask=input_mask,
                                                                   return_dict_in_generate=True,
                                                                   num_return_sequences=self.configs.num_beams,
                                                                   max_length=self.configs.eval_tgt_max_length,
                                                                   diversity_penalty=diversity_penalty,
                                                                   num_beam_groups=num_beam_groups,
                                                                   prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                                                   num_beams=self.configs.num_beams,
                                                                   output_scores=True,
                                                                   bos_token_id=0,)
            else:
                outputs = self.T5ForConditionalGeneration.generate(input_ids=src_ids,
                                                                   attention_mask=src_mask,
                                                                   return_dict_in_generate=True,
                                                                   num_return_sequences=self.configs.num_beams,
                                                                   max_length=self.configs.eval_tgt_max_length,
                                                                   diversity_penalty=diversity_penalty,
                                                                   num_beam_groups=num_beam_groups,
                                                                   prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                                                   num_beams=self.configs.num_beams,
                                                                   output_scores=True,
                                                                   entity_hidden_state=entity_hidden_state,
                                                                   entity_mask = None,
                                                                   addsource=addsource,
                                                                   sep =  sep,
                                                                   )
            raw_generated_text = self.trainer.datamodule.tokenizer.batch_decode(outputs.sequences)
            generated_text = _extract(raw_generated_text)#320个，实体text
            # print(outputs.sequences_scores)
            assert len(generated_text) == self.configs.num_beams * len(src_ids)
            return generated_text,outputs.sequences_scores
        elif self.configs.decoder == 'do_sample':
            if self.configs.use_soft_prompt:
                # input_index .shape: (batch_size, seq_len + 4)
                input_index = batched_data['input_index']
                # soft_prompt_index .shape: (batch_size, 4)
                soft_prompt_index = batched_data['soft_prompt_index']
                inputs_emb, input_mask = self.get_soft_prompt_input_embed(src_ids, src_mask, self.ent_rel, input_index,
                                                                          soft_prompt_index)
                outputs = self.T5ForConditionalGeneration.generate(inputs_embeds=inputs_emb,
                                                                   attention_mask=input_mask,
                                                                   return_dict_in_generate=True,
                                                                   num_return_sequences=self.configs.num_beams,
                                                                   max_length=self.configs.eval_tgt_max_length,
                                                                   output_scores=True,
                                                                   do_sample=True)
            else:
                outputs = self.T5ForConditionalGeneration.generate(input_ids=src_ids,
                                                                   attention_mask=src_mask,
                                                                   return_dict_in_generate=True,
                                                                   num_return_sequences=self.configs.num_beams,
                                                                   max_length=self.configs.eval_tgt_max_length,
                                                                   output_scores=True,
                                                                   do_sample=True)

            raw_generated_text = self.trainer.datamodule.tokenizer.batch_decode(outputs.sequences)
            generated_text = _extract(raw_generated_text)
            assert len(generated_text) == self.configs.num_beams * len(src_ids)
            # sequences .shape: (batch_size * num_beams, max_seq_len - 1)
            sequences = outputs.sequences[:, 1:]
            # scores: .shape: (batch_size * num_beams, max_seq_len - 1, vocab_size)
            scores = torch.stack(outputs.scores).transpose(0, 1)
            # scores: .shape: (batch_size * num_beams, max_seq_len - 1, 1)
            scores = torch.stack([torch.gather(scores[i], 1, sequences[i].unsqueeze(1)) for i in range(len(sequences))])
            # scores .type: list(float) .len: batch_size * num_beams
            scores = torch.mean(scores.squeeze(-1), dim=-1).tolist()

            new_generated_text = []
            for i in range(0, len(generated_text), self.configs.num_beams):
                gen_seqs = generated_text[i:i + self.configs.num_beams]
                gen_scores = scores[i:i + self.configs.num_beams]
                gen_seqs = [seq for _, seq in sorted(list(zip(gen_scores, gen_seqs)), key=lambda x: x[0], reverse=True)]
                new_generated_text.extend(gen_seqs)
            return new_generated_text
        else:
            raise ValueError('Invalid decoder')

    def get_soft_prompt_input_embed(self, src_ids, src_mask, ent_rel, input_index, soft_prompt_index):
        # ent_ids, rel_ids .shape: (batch_size, 1)
        ent_ids, rel_ids = ent_rel[:, [0]], ent_rel[:, [1]]
        # ent_emb1, ent_emb2, rel_emb1, rel_emb2 .shape: (batch_size, 1, model_dim)
        if self.configs.use_rel_prompt_emb:
            ent_emb1, ent_emb2 = self.rel_embed3(rel_ids), self.rel_embed4(rel_ids)
        else:
            ent_emb1, ent_emb2 = self.ent_embed1(ent_ids), self.ent_embed2(ent_ids)
        rel_emb1, rel_emb2 = self.rel_embed1(rel_ids), self.rel_embed2(rel_ids)
        # ent_emb, rel_emb .shape: (batch_size, 2, model_dim)
        ent_emb, rel_emb = torch.cat([ent_emb1, ent_emb2], dim=1), torch.cat([rel_emb1, rel_emb2], dim=1)
        # soft_prompt_emb .shape: (batch_size, 4, model_dim)
        soft_prompt_emb = torch.cat([ent_emb, rel_emb], dim=1)
        # inputs_emb .shape: (batch_size, seq_len, model_dim)
        inputs_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(src_ids)
        batch_size, seq_len, model_dim = inputs_emb.shape
        # indicator_in_batch .shape: (batch_size, 1) .examples: torch.LongTensor([[0], [1], [2], [3]])
        indicator_in_batch = torch.arange(batch_size).type_as(ent_ids).unsqueeze(-1)

        # inputs_emb .shape: (batch_size * seq_len, model_dim)
        inputs_emb = inputs_emb.view(-1, model_dim)#64*62,768
        input_index = (input_index + indicator_in_batch * seq_len).view(-1)#数字以62为一个隔阂，64*66
        # inputs_emb .shape: (batch_size * (seq_len + 4), model_dim)
        inputs_emb = torch.index_select(inputs_emb, 0, input_index)#4224(64*66),768，就是在这一步给扩展了input_emb
        soft_prompt_index = (soft_prompt_index + indicator_in_batch * (seq_len + 4)).view(-1)
        inputs_emb[soft_prompt_index] = soft_prompt_emb.view(batch_size * 4, model_dim)
        inputs_emb = inputs_emb.view(batch_size, -1, model_dim)#64,66,768

        input_mask = torch.cat([torch.ones(batch_size, 4).type_as(src_mask), src_mask], dim=1)
        return inputs_emb, input_mask

    def log_generation(self, group_text, src_names, target_names, batch_idx, dataset_idx):
        log_file = os.path.join(self.configs.save_dir, 'Epoch-' + str(self.current_epoch) + '-generation.tmp')
        with open(log_file, 'a', encoding='utf-8') as file:
            for i, texts in enumerate(group_text):
                file.write(str(batch_idx * self.configs.val_batch_size + i) + ' -- ' + src_names[i] + ' => ' + target_names[i] + '\n')
                if self.configs.temporal:
                    hr_key = (self.test_triple[i][dataset_idx], self.test_triple[i][2], self.test_triple[i][3])
                else:
                    hr_key = (self.test_triple[i][dataset_idx], self.test_triple[i][2])
                all_gt_ids = self.all_ground_truth[hr_key]
                all_gt_seqs = [self.ent_name_list[ids] for ids in all_gt_ids]
                ii = 1
                for text_i, text in enumerate(texts):
                    if text == target_names[i]:
                        file.write('\t\t%2d %10s %s\n' % (ii, '(target):', text))
                        ii += 1
                    elif text in all_gt_seqs:
                        file.write('\t\t%2s %10s %s\n' % ('', '', text))
                    elif text in self.ent_name_list:
                        file.write('\t\t%2d %10s %s\n' % (ii, '(ent):', text))
                        ii += 1
                    else:
                        file.write('\t\t%2d %10s %s\n' % (ii, '(non-ent):', text))
                        ii += 1

    def validation_epoch_end(self, outs):
        if self.pretrainKG:
            if len(outs)!=0:
                agg_out = dict()
                for out in outs:
                    for key, value in out.items():
                        if key in agg_out:
                            agg_out[key] += value
                        else:
                            agg_out[key] = value
                self.log('loss', agg_out['loss'].item())
                print(agg_out['loss'].item())
        else:
            if self.current_epoch < self.configs.skip_n_val_epoch:
                return
            pred_tail_out, pred_head_out = outs
            agg_tail_out, agg_head_out, agg_total_out = dict(), dict(), dict()
            for out in pred_tail_out:
                for key, value in out.items():
                    if key in agg_tail_out:
                        agg_tail_out[key] += value
                    else:
                        agg_tail_out[key] = value

            for out in pred_head_out:
                for key, value in out.items():
                    if key in agg_head_out:
                        agg_head_out[key] += value
                    else:
                        agg_head_out[key] = value

            tail_ranks, head_ranks = agg_tail_out['ranks'], agg_head_out['ranks']
            del agg_tail_out['ranks']
            del agg_head_out['ranks']

            perf = get_performance(self, tail_ranks, head_ranks)
            print(perf)

    def test_step(self, batched_data, batch_idx, dataset_idx):
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        if self.configs.optim == 'Adafactor':
            optim = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.configs.lr)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
        return optim



class N3(object):
    def __init__(self, lmbda: float):
        super(N3, self).__init__()
        self.lmbda = lmbda

    def penalty(self, factors):
        """

        :param factors: tuple, (s, p, o), batch_size * rank
        :return:
        """
        norm, raw = 0, 0
        for f in factors:
            norm += self.lmbda * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]

    def checkpoint(self, regularizer_cache_path, epoch_id):
        if regularizer_cache_path is not None:
            print('Save the regularizer at epoch {}'.format(epoch_id))
            path = regularizer_cache_path + '{}.reg'.format(epoch_id)
            torch.save(self.state_dict(), path)
            print('Regularizer Checkpoint:{}'.format(path))