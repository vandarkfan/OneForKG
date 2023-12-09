import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from helper import batchify, get_soft_prompt_pos

from random import sample, randint, random
class TrainDataset(Dataset):
    def __init__(self, configs, tokenizer, train_triples, name_list_dict, prefix_trie_dict, ground_truth_dict,mode):
        self.configs = configs
        self.train_triples = train_triples
        self.tokenizer = tokenizer
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.src_description_list = name_list_dict['src_description_list']
        self.tgt_description_list = name_list_dict['tgt_description_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.neg_candidate_mask = prefix_trie_dict['neg_candidate_mask']
        self.mode = mode

    def __len__(self):
        # return len(self.train_triples) * 2
        return len(self.train_triples)
    def __getitem__(self, index):
        # train_triple = self.train_triples[index // 2]
        # mode = 'tail' if index % 2 == 0 else 'head'
        train_triple = self.train_triples[index]
        mode = self.mode
        if self.configs.temporal:
            head, tail, rel, time = train_triple
        else:
            head, tail, rel = train_triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]
        if self.configs.src_descrip_max_length > 0:
            head_descrip, tail_descrip = '[' + self.src_description_list[head] + ']', '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''
        if self.configs.tgt_descrip_max_length > 0:
            head_target_descrip, tail_target_descrip = '[' + self.tgt_description_list[head] + ']', '[' + self.tgt_description_list[tail] + ']'
        else:
            head_target_descrip, tail_target_descrip = '', ''

        if mode == 'tail':
            if self.configs.temporal:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | ' + time
            else:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>'
            tgt = '<extra_id_0>' + tail_name + tail_target_descrip + '<extra_id_1>'
        else:
            if self.configs.temporal:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip + ' | ' + time
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip
            tgt = '<extra_id_0>' + head_name + head_target_descrip + '<extra_id_1>'


        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(tgt, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask
        target_ent = torch.tensor(tail)
        if mode == 'tail':
            target_ent = torch.tensor(tail)
        if mode == 'head':
            target_ent = torch.tensor(head)
        ent_rel = torch.LongTensor([head, rel]) if mode == 'tail' else torch.LongTensor([tail, rel])

        if self.configs.temporal:
            sep1, sep2, sep3 = [ids for ids in range(len(source_ids)) if source_ids[ids] == 1820]
            # if mode == 'tail':
            #     input_index = [0] + list(range(0, sep1)) + [0] + [sep1] + [0] + list(range(sep1 + 1, sep2)) + [
            #         0] + list(range(sep2, len(source_ids)))
            #     soft_prompt_index = torch.LongTensor([0, sep1 + 1, sep1 + 3, sep2 + 3])
            # elif mode == 'head':
            #     input_index = list(range(0, sep1 + 1)) + [0] + list(range(sep1 + 1, sep2)) + [0, sep2, 0] + list(
            #         range(sep2 + 1, sep3)) + [0] + list(range(sep3, len(source_ids)))
            #     soft_prompt_index = torch.LongTensor([sep2 + 3, sep3 + 3, sep1 + 1, sep2 + 1])
        else:
            sep1, sep2 = [ids for ids in range(len(source_ids)) if source_ids[ids] == 1820]
            sep3 = -1
            if mode == 'head':
                sep3 = len(source_ids)
        sep = torch.LongTensor([sep1, sep2, sep3])
        out = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
                'train_triple': train_triple,
                'ent_rel': ent_rel,
                'mode':mode,
                'target_ent': target_ent,
                'sep': sep,
        }

        if self.configs.use_soft_prompt:
            input_index, soft_prompt_index, target_soft_prompt_index = get_soft_prompt_pos(self.configs, source_ids, target_ids, mode)
            out['input_index'] = input_index
            out['soft_prompt_index'] = soft_prompt_index
            out['target_soft_prompt_index'] = target_soft_prompt_index
        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['train_triple'] = batchify(data, 'train_triple', return_list=True)
        agg_data['ent_rel'] = batchify(data, 'ent_rel')
        agg_data['mode'] = [out['mode'] for out in data]
        agg_data['target_ent'] = [out['target_ent'] for out in data]
        agg_data['sep'] = [out['sep'] for out in data]

        if self.configs.use_soft_prompt:
            agg_data['input_index'] = batchify(data, 'input_index', padding_value=0)
            agg_data['soft_prompt_index'] = batchify(data, 'soft_prompt_index')
            agg_data['target_soft_prompt_index'] = batchify(data, 'target_soft_prompt_index')
        return agg_data


class TestDataset(Dataset):
    def __init__(self, configs, tokenizer, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict, mode):  # mode: {tail, head}
        self.configs = configs
        self.test_triples = test_triples
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.src_description_list = name_list_dict['src_description_list']
        self.tgt_description_list = name_list_dict['tgt_description_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.test_triples)

    def __getitem__(self, index):
        test_triple = self.test_triples[index]
        if self.configs.temporal:
            head, tail, rel, time = test_triple
        else:
            head, tail, rel = test_triple
        head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]

        if self.configs.src_descrip_max_length > 0:
            head_descrip, tail_descrip = '[' + self.src_description_list[head] + ']', '[' + self.src_description_list[tail] + ']'
        else:
            head_descrip, tail_descrip = '', ''

        if self.mode == 'tail':
            if self.configs.temporal:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | ' + time
            else:
                src = head_name + ' ' + head_descrip + ' | ' + rel_name + ' | ' + '<extra_id_0>'
            tgt_ids = tail
        else:
            if self.configs.temporal:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip + ' | ' + time
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' ' + tail_descrip
            tgt_ids = head

        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        source_names = src
        target_names = self.ent_name_list[tgt_ids]

        # ent_rel = test_triple[[0, 2]] if self.mode == 'tail' else test_triple[[1, 2]]
        ent_rel = torch.LongTensor([head, rel]) if self.mode == 'tail' else torch.LongTensor([tail, rel])

        target_ent = torch.tensor(tail)
        if self.mode == 'tail':
            target_ent = torch.tensor(tail)
        if self.mode == 'head':
            target_ent = torch.tensor(head)

        if self.configs.temporal:
            sep1, sep2, sep3 = [ids for ids in range(len(source_ids)) if source_ids[ids] == 1820]
            # if mode == 'tail':
            #     input_index = [0] + list(range(0, sep1)) + [0] + [sep1] + [0] + list(range(sep1 + 1, sep2)) + [
            #         0] + list(range(sep2, len(source_ids)))
            #     soft_prompt_index = torch.LongTensor([0, sep1 + 1, sep1 + 3, sep2 + 3])
            # elif mode == 'head':
            #     input_index = list(range(0, sep1 + 1)) + [0] + list(range(sep1 + 1, sep2)) + [0, sep2, 0] + list(
            #         range(sep2 + 1, sep3)) + [0] + list(range(sep3, len(source_ids)))
            #     soft_prompt_index = torch.LongTensor([sep2 + 3, sep3 + 3, sep1 + 1, sep2 + 1])
        else:
            sep1, sep2 = [ids for ids in range(len(source_ids)) if source_ids[ids] == 1820]
            sep3 = -1
            if self.mode == 'head':
                sep3 = len(source_ids)
        sep = torch.LongTensor([sep1, sep2, sep3])
        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'source_names': source_names,
            'target_names': target_names,
            'test_triple': test_triple,
            'ent_rel': ent_rel,
            'mode': self.mode,
            'target_ent': target_ent,
            'sep': sep,
        }
        if self.configs.use_soft_prompt:
            input_index, soft_prompt_index, _ = get_soft_prompt_pos(self.configs, source_ids, None, self.mode)
            out['input_index'] = input_index
            out['soft_prompt_index'] = soft_prompt_index
        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['source_names'] = [dt['source_names'] for dt in data]
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        agg_data['test_triple'] = batchify(data, 'test_triple', return_list=True)
        agg_data['ent_rel'] = batchify(data, 'ent_rel')
        agg_data['mode'] = [out['mode'] for out in data]
        agg_data['target_ent'] = [out['target_ent'] for out in data]
        agg_data['sep'] = [out['sep'] for out in data]
        if self.configs.use_soft_prompt:
            agg_data['input_index'] = batchify(data, 'input_index', padding_value=0)
            agg_data['soft_prompt_index'] = batchify(data, 'soft_prompt_index')
        return agg_data
class PretrainTrainDataset(Dataset):
    def __init__(self, configs, tokenizer, train_list, name_list_dict, ground_truth_dict):
        self.configs = configs
        self.train_list = train_list
        self.tokenizer = tokenizer
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']


    def __len__(self):
        # return len(self.train_triples) * 2
        return len(self.train_list)
    def __getitem__(self, index):
        # train_triple = self.train_triples[index // 2]
        # mode = 'tail' if index % 2 == 0 else 'head'
        instance = self.train_list[index]
        tokens_a, tokens_b, input_entity_id, word_subword = instance#poc是进到这里面了
        pad_number = len(self.ent_name_list)
        input_entity_id = input_entity_id + [pad_number]

        src = ' '.join(tokens_a)
        tgt = ' '.join(tokens_b)
        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(tgt, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask

        n_pad = len(source_ids) - len(input_entity_id)
        word_mask = [1] * len(input_entity_id)
        word_mask.extend([0] * n_pad)
        input_entity_id.extend([pad_number] * n_pad)
        out = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
                'word_mask': word_mask,
                'input_entity_id': input_entity_id,
        }

        return out

    def collate_fn(self, data):
        pad_number = len(self.ent_name_list)
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['word_mask'] = batchify(data, 'word_mask', padding_value=0)
        agg_data['input_entity_id'] = batchify(data, 'input_entity_id', padding_value=pad_number)
        if self.configs.use_soft_prompt:
            agg_data['input_index'] = batchify(data, 'input_index', padding_value=0)
            agg_data['soft_prompt_index'] = batchify(data, 'soft_prompt_index')
            agg_data['target_soft_prompt_index'] = batchify(data, 'target_soft_prompt_index')
        return agg_data


class PretrainTestDataset(Dataset):
    def __init__(self, configs, tokenizer, test_list, name_list_dict, ground_truth_dict):  # mode: {tail, head}
        self.configs = configs
        self.test_list = test_list
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.test_list)

    def __getitem__(self, index):
        instance = self.test_list[index]
        tokens_a, tokens_b, input_entity_id, word_subword = instance  # poc是进到这里面了
        pad_number = len(self.ent_name_list)
        input_entity_id = input_entity_id + [pad_number]

        src = ' '.join(tokens_a)
        tgt = ' '.join(tokens_b)
        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(tgt, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask

        n_pad = len(source_ids) - len(input_entity_id)
        word_mask = [1] * len(input_entity_id)
        word_mask.extend([0] * n_pad)
        input_entity_id.extend([pad_number] * n_pad)
        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'word_mask': word_mask,
            'input_entity_id': input_entity_id,
        }

        return out

    def collate_fn(self, data):
        pad_number = len(self.ent_name_list)
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['word_mask'] = batchify(data, 'word_mask', padding_value=0)
        agg_data['input_entity_id'] = batchify(data, 'input_entity_id', padding_value=pad_number)
        if self.configs.use_soft_prompt:
            agg_data['input_index'] = batchify(data, 'input_index', padding_value=0)
            agg_data['soft_prompt_index'] = batchify(data, 'soft_prompt_index')
        return agg_data

class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, configs, name_list_dict, ground_truth_dict):
        super().__init__()
        self.configs = configs
        # ent_name_list, rel_name_list .type: list
        self.name_list_dict = name_list_dict
        self.ground_truth_dict = ground_truth_dict

        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        # self.train_both = None
        self.train= None
        self.valid= None
        self.test= None
        self.ex_list = []
        self.pretraining_num=50000
        all_entity = self.name_list_dict['ent_name_list']
        entity_id = self.name_list_dict['entname2id']
        while len(self.ex_list) < self.pretraining_num:  # 一共要400000次
            src = sample(all_entity, 5)  # 随便找了五个实体
            mask_num = randint(1,5)
            src_entity_id = []
            word_subword = []
            src_tk = []
            tgt_tk = []
            masked_num = 0
            for e_i, entity in enumerate(src):
                # entity = entity.split(',')[0]
                entity_tk = self.tokenizer.tokenize(entity)  # 分解成小的piece
                src_entity_id.append(entity_id[entity])  # 实体id往里面去
                word_subword.append(len(entity_tk))
                if random() >= 0.5 and mask_num - masked_num > 0:
                    src_tk.append('<extra_id_'+ str(masked_num) +'>')
                    if masked_num==0:
                        tgt_tk.append('<extra_id_'+ str(masked_num) +'> ' + entity + '<extra_id_'+ str(masked_num + 1) +'> ')
                    else:
                        tgt_tk.append( entity + '<extra_id_' + str(masked_num + 1) + '> ')
                    masked_num += 1
                else:
                    src_tk.append(entity)

                    # for tk in entity_tk:
                    #     src_tk.append(tk)
                    #     tgt_tk.append(tk)
            self.ex_list.append((src_tk, tgt_tk, src_entity_id, word_subword))
        self.train_list = self.ex_list[:40000]
        self.valid_list = self.ex_list[40000:45000]
        self.test_list = self.ex_list[45000:]
    def prepare_data(self):
        self.train = PretrainTrainDataset(self.configs, self.tokenizer, self.train_list, self.name_list_dict, self.ground_truth_dict,)
        self.valid = PretrainTestDataset(self.configs, self.tokenizer, self.valid_list, self.name_list_dict, self.ground_truth_dict, )
        self.test = PretrainTestDataset(self.configs, self.tokenizer, self.test_list, self.name_list_dict, self.ground_truth_dict, )
    def train_dataloader(self):
        train_loader = DataLoader(self.train,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)

        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)

        return test_loader
class DataModule(pl.LightningDataModule):
    def __init__(self, configs, train_triples, valid_triples, test_triples, name_list_dict, prefix_trie_dict, ground_truth_dict):
        super().__init__()
        self.configs = configs
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        # ent_name_list, rel_name_list .type: list
        self.name_list_dict = name_list_dict
        self.prefix_trie_dict = prefix_trie_dict
        self.ground_truth_dict = ground_truth_dict

        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        # self.train_both = None
        self.train_tail, self.train_head = None, None
        self.valid_tail, self.valid_head = None, None
        self.test_tail, self.test_head = None, None

    def prepare_data(self):
        # self.train_both = TrainDataset(self.configs, self.tokenizer, self.train_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict)
        self.train_tail = TrainDataset(self.configs, self.tokenizer, self.train_triples, self.name_list_dict,self.prefix_trie_dict, self.ground_truth_dict,'tail')
        self.train_head = TrainDataset(self.configs, self.tokenizer, self.train_triples, self.name_list_dict,self.prefix_trie_dict, self.ground_truth_dict, 'head')
        self.valid_tail = TestDataset(self.configs, self.tokenizer, self.valid_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'tail')
        self.valid_head = TestDataset(self.configs, self.tokenizer, self.valid_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'head')
        self.test_tail = TestDataset(self.configs, self.tokenizer, self.test_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'tail')
        self.test_head = TestDataset(self.configs, self.tokenizer, self.test_triples, self.name_list_dict, self.prefix_trie_dict, self.ground_truth_dict, 'head')

    def train_dataloader(self):
        # train_loader = DataLoader(self.train_both,
        #                           batch_size=self.configs.batch_size,
        #                           shuffle=True,
        #                           collate_fn=self.train_both.collate_fn,
        #                           pin_memory=True,
        #                           num_workers=self.configs.num_workers)
        train_tail_loader = DataLoader(self.train_tail,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_tail.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        train_head_loader = DataLoader(self.train_head,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_head.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return [train_tail_loader, train_head_loader]

    def val_dataloader(self):
        valid_tail_loader = DataLoader(self.valid_tail,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_tail.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        valid_head_loader = DataLoader(self.valid_head,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_head.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        return [valid_tail_loader, valid_head_loader]

    def test_dataloader(self):
        test_tail_loader = DataLoader(self.test_tail,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_tail.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        test_head_loader = DataLoader(self.test_head,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_head.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        return [test_tail_loader, test_head_loader]
