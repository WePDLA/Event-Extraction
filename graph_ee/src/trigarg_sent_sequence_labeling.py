# Copyright Kan Zhigang
#
"""
sequence labeling
"""

print("Loading packages ...")
import ast
import os
import json
import warnings
import random
import argparse
import tqdm
from functools import partial
from transformers import BertModel, BertTokenizer, AdamW, PreTrainedTokenizerFast, AutoTokenizer
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
from module.trig_BERTLiner_model import Trig_BertLiner_Model
from module.trig_BERTCRF_model import Trig_BertCRF_Model
from module.trig_BERTMultAtt_model import Trig_BertMultAtt_Model
print("Load packages successfully")


# # yapf: disable
# parser = argparse.ArgumentParser(__doc__)
# parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
# parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
# parser.add_argument("--tag_path", type=str, default=None, help="tag set path")
# parser.add_argument("--vocab_path", type=str, default=None, help="vocab path")
# parser.add_argument("--train_data", type=str, default=None, help="train data")
# parser.add_argument("--dev_data", type=str, default=None, help="dev data")
# parser.add_argument("--test_data", type=str, default=None, help="test data")
# parser.add_argument("--predict_data", type=str, default=None, help="predict data")
# parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
# parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
# parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
# parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
# parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
# parser.add_argument("--valid_step", type=int, default=100, help="validation step")
# parser.add_argument("--skip_step", type=int, default=20, help="skip step")
# parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
# parser.add_argument("--checkpoints", type=str, default=None, help="Directory to model checkpoint")
# parser.add_argument("--init_ckpt", type=str, default=None, help="already pretraining model checkpoint")
# parser.add_argument("--predict_save_path", type=str, default=None, help="predict data save path")
# parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
# parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use, 0 for CPU.")
# args = parser.parse_args()
# yapf: enable.

# commen parms
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 2e-5
epochs = 50
example_mark = False

batch_size = 16
config_dir = './conf/DuEE/'
data_dir = '../../data/DuEE/'
model_save_dir = '../../../save/DuEE/'
model_name = 'bert-base-chinese'
# MODEL_PATH = '../../../bert_models/bert-base-chinese/'
# MODEL_PATH = '../../../bert_models/roberta-wwm-large/'
MODEL_PATH = '../../../bert_models/ernie-1.0/'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

task_name = 'sent'
bert_model_name = 'ernie'
tedian = 'lr-2'
# bert_model_name = 'robert-large'
model_type = 'crf'
model_type = '{}_{}_{}_{}'.format(task_name,bert_model_name,tedian,model_type)
# model_type = 'sent_bert_liner'
# model_type = 'sent_bert_crf'
# model_type = 'sent_bert_multatt'
# model_type = 'sent_ernie_crf'
# model_type = 'sent_ernie_liner'
f1_first, acc_first  = True, False

print('model_type:',model_type)

#trig parms
do_train_mark = True
do_predict_mark = False


#arg parms
do_arg_train_mark = True
do_arg_predict_mark = False


print("load tokenizer done")

def load_dict_test(dict_path):
    id2label = {0:'O'}
    label2id = {'O':0}
    # id2label = {0:'O', -100:'PAD'}
    # label2id = {'O':0,'PAD':-100}

    with open(dict_path, 'r')as f1:
        lines = f1.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            label = line.split('\t')[1]
            label = label[0]
            if label not in label2id.keys():
                # new_id = len(label2id)
                ## 因为'PAD'在字典中占了位置，所以计算长度时要去掉'PAD'
                new_id = len(label2id)
                label2id[label] = new_id
                id2label[new_id] = label
    return id2label, label2id

def load_dataset_test(data_path, label2id, id2label, dataset_type = 'not_train'):
    word_list = []
    label_list = []
    # label_id_list = []
    with open(data_path, 'r', encoding='utf-8') as fp:
        # skip the head line
        next(fp)
        for line in fp.readlines():
            words, labels = line.strip('\n').split('\t')
            words = words.split('\002')
            labels = labels.split('\002')
            label_test_list = []
            # label_ids = []
            for label in labels:
                label = label[0]
                label_test_list.append(label)
                if label not in label2id.keys() and dataset_type == 'train':
                    # new_id = len(label2id)
                    new_id = len(label2id)
                    label2id[label] = new_id
                    id2label[new_id] = label
                # label_ids.append(label2id[label])
            words_cleaned = sentence_clean(words)
            word_list.append(words_cleaned)
            label_list.append(label_test_list)
            # label_id_list.append(label_ids)
    return word_list, label_list, label2id, id2label

def load_dict(dict_path):
    id2label = {0: 'O'}
    label2id = {'O': 0}
    # id2label = {}
    # label2id = {}
    with open(dict_path, 'r')as f1:
        lines = f1.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            # id = int(line.split('\t')[0])
            label = line.split('\t')[1]
            if label not in label2id.keys():
                new_id = len(label2id)
                id2label[new_id] = label
                label2id[label] = new_id
    return id2label, label2id

def write_dict2file(id2labels, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'a+')as f1:
        for id in id2labels.keys():
            item = str(id) + '\t' + id2labels[id] + '\n'
            f1.write(item)

def convert_example_to_feature(example, tokenizer, label_vocab=None, max_seq_len=512, no_entity_label="O", ignore_label=-1, is_test=False):
    tokens, labels = example
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        pad_to_max_seq_len=True,
        max_seq_len=max_seq_len)

    input_ids = tokenized_input[0]['input_ids']
    token_type_ids = tokenized_input[0]['token_type_ids']
    seq_len = tokenized_input[0]['seq_len']

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels = labels[:(max_seq_len-2)]
        encoded_label = [no_entity_label] + labels + [no_entity_label]
        encoded_label = [label_vocab[x] for x in encoded_label]
        # padding label
        encoded_label += [ignore_label]* (max_seq_len - len(encoded_label))
        return input_ids, token_type_ids, seq_len, encoded_label

def sentence_clean(word_list):
    durty_char_list = ['\u3000','\u2003','\u2002', '\ufeff', '\u200b', '\xa0','\ue627',
                       '，']
    # durty_char_list = ['\u3000', '\u2003', '\u2002', '\ufeff', '\u200b', '\xa0', '\ue627',
                       # ' ']
    words_cleaned_list = ['*' if i in durty_char_list else i for i in word_list]
    return words_cleaned_list

def load_dataset(data_path, label2id, id2label, dataset_type = 'not_train', is_predict=False):
    word_list = []
    label_list = []
    # label_id_list = []
    if not is_predict:
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                # label_ids = []
                for label in labels:
                    if label not in label2id.keys() and dataset_type == 'train':
                        new_id = len(label2id)
                        label2id[label] = new_id
                        id2label[new_id] = label
                    # label_ids.append(label2id[label])
                words_cleaned = sentence_clean(words)
                word_list.append(words_cleaned)
                label_list.append(labels)
                # label_id_list.append(label_ids)
    else:
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                words = line.strip('\n')
                words = words.split('\002')
                words_cleaned = sentence_clean(words)
                labels = ['O'] * len(words_cleaned)
                word_list.append(words_cleaned)
                label_list.append(labels)
    return word_list, label_list, label2id, id2label

class EE_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, senq_len_list):
        self.encodings = encodings
        self.labels = labels
        self.senq_len = senq_len_list
        # print('done')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['senq_len'] = torch.tensor(self.senq_len[idx])
        return item

    def __len__(self):
        return len(self.labels)

def encode_tags(tags, encodings, trig_label2id, train_texts=[], special_char_list = []):
    labels = [[trig_label2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    senq_len_list = []
    count = 0
    for doc_labels, doc_offset, text, token_ids in zip(labels, encodings.offset_mapping,
                                train_texts, encodings.input_ids):
        # create an empty array of -100
        if len(doc_labels) > 510:
            doc_labels = doc_labels[:510]
        senq_len = len(doc_labels) + 2
        ## 在label list的头部和尾部加上'[CLS]'和'[SEP]'对应的标签
        # doc_labels = [trig_label2id['O']] + doc_labels + [trig_label2id['O']]
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * trig_label2id['PAD']
        # doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0

        try:
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        except:
            for char in text:
                a = tokenizer([char]).input_ids[0]
                # b = len(a)
                if len(a) <= 2:
                    special_char_list.append(char)
                # print(len(a))
            # doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels

        ## 将'[CLS]'和'[SEP]'对应的doc_enc_labels赋值为'O'
        doc_enc_labels[0] = trig_label2id['O']
        doc_enc_labels[len(doc_labels) + 1] = trig_label2id['O']
        encoded_labels.append(doc_enc_labels.tolist())
        senq_len_list.append(senq_len)
        count += 1
    return encoded_labels, special_char_list, senq_len_list

def load_DataAndLabelDict(data_file, data_id2label, data_label2id ,tokenizer,
                          dataset_type='not_train', special_char_list=[],
                          is_predict=False,mode = 'trig'):
    # 读取训练文件，获得文本及其对应的标签，对于样例数据标签类型不全的问题，补全标签字典
    data_texts, data_labels, data_label2id, data_id2label = \
    load_dataset(data_file, data_label2id, data_id2label,
        dataset_type=dataset_type, is_predict=is_predict)
    # data_texts, data_labels, data_label2id, data_id2label = \
    # load_dataset_test(data_data_file, data_label2id, data_id2label, dataset_type = 'train')

    if 'PAD' not in data_label2id.keys():
        new_id = len(data_label2id)
        data_label2id['PAD'] = new_id
        data_id2label[new_id] = 'PAD'

    # 对文本进行tokenizer
    data_encodings = tokenizer(data_texts, is_split_into_words=True,
            return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    # 对标签进行编码（主要是将其补充到与tokenizer文本一样的长度）
    data_encoded_labels, special_char_list, senq_len_list = encode_tags(data_labels, data_encodings, data_label2id,
                                       data_texts, special_char_list=special_char_list)
    train_dataset = EE_Dataset(data_encodings, data_encoded_labels, senq_len_list)
    if is_predict:
        shuffle = False
    else:
        shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_label2id, data_id2label, train_loader, special_char_list

def do_trig_train(train_data_file,dev_data_file,test_data_file,dict_file,dict_save_file,
                  dee_model_name, model_type='bert_liner',mode='trig'):
    # load datasets
    print("Loading datasets ...")
    # 读标签和id的对应文件
    if os.path.exists(dict_save_file):
        data_id2label, data_label2id = load_dict(dict_save_file)
    else:
        data_id2label, data_label2id = load_dict(dict_file)
    special_char_list = []
    data_label2id, data_id2label, train_loader, special_char_list = load_DataAndLabelDict(train_data_file,
                data_id2label, data_label2id, tokenizer, dataset_type='train',
                special_char_list=special_char_list, mode=mode)
    data_label2id, data_id2label, dev_loader, special_char_list = load_DataAndLabelDict(dev_data_file,
                data_id2label, data_label2id, tokenizer, dataset_type='train',
                special_char_list=special_char_list, mode=mode)
    if not os.path.exists(dict_save_file):
        write_dict2file(data_id2label, dict_save_file)
    if len(special_char_list) > 0:
        raise NotImplementedError('Find special char!')
    #定义模型所需参数
    ## 减1是为了去掉‘PAD’
    label_num = len(data_id2label)
    print("Creating model ...")
    # my model
    labels_need = list(data_id2label.keys())
    O_index = labels_need.index(data_label2id['O'])
    labels_need.pop(O_index)
    PAD_index = labels_need.index(data_label2id['PAD'])
    labels_need.pop(PAD_index)
    if model_type.lower().split('_')[1][-5:] == 'large':
        hidden_size = 1024
    else:
        hidden_size = 768
    # config = AutoConfig.from_pretrained(MODEL_PATH)
    if model_type.lower().split('_')[-1] == 'liner':
        model = Trig_BertLiner_Model(MODEL_PATH,tokenizer,label_num,
                hidden_size=hidden_size, label2id = data_label2id)
    elif model_type.lower().split('_')[-1] == 'crf':
        label_num = len(data_id2label) - 1
        model = Trig_BertCRF_Model(MODEL_PATH, tokenizer, label_num,
                hidden_size=hidden_size,device= device)
    elif model_type.lower().split('_')[-1] == 'multatt':
        model = Trig_BertMultAtt_Model(MODEL_PATH, tokenizer, label_num,
                head_num=12, att_output_dim = 96, device= device)
    else:
        raise NotImplementedError("没有定义模型")
    model.to(device)
    optim = AdamW(model.parameters(), lr=learning_rate)
    print("training ...")
    #定义模型的最好性能指标
    if os.path.exists(dee_model_name):
        model.load_state_dict(torch.load(dee_model_name))
        p_tmp, r_tmp, best_f1, best_acc = model.dev_part(model, dev_loader, labels_need, device)
        print('p:{}  r:{}  f1:{}  best_acc:{}'.format(p_tmp, r_tmp, best_f1, best_acc))
    else:
        best_f1 = 0.
        best_acc = 0.
    #开始训练
    for epoch in range(epochs):
        model.train_part(model, train_loader, optim, device)
        p_dev, r_dev, f1_dev, acc_dev = model.dev_part(model, dev_loader, labels_need, device)
        #保存模型
        if f1_dev > best_f1:
            best_f1 = f1_dev
            if f1_first:
                torch.save(model.state_dict(), dee_model_name)
        if acc_dev > best_acc:
            best_acc = acc_dev
            if acc_first:
                torch.save(model.state_dict(), dee_model_name)
        print('{}/{}\tp:{}  r:{}  f1:{}  acc:{}\nbest_f1:{}  best_acc:{}'
              .format(epoch, epochs, p_dev, r_dev, f1_dev, acc_dev, best_f1, best_acc))

def do_trig_predict(test_data_file,pred_sents_file, dict_file,dict_save_file, dee_model_name,
                    predict_save_path, mode='trig'):
    # test_data_file = trig_data_dir + 'test.tsv'
    print("Loading datasets ...")
    if os.path.exists(dict_save_file):
        data_id2label, data_label2id = load_dict(dict_save_file)
    else:
        data_id2label, data_label2id = load_dict(dict_file)

    # data_id2label, data_label2id = load_dict(dict_file)
    # trig_id2label, trig_label2id = load_dict_test(trigger_dict_file)
    special_char_list = []
    data_label2id, data_id2label, test_loader, special_char_list = load_DataAndLabelDict(test_data_file,
        data_id2label, data_label2id, tokenizer, special_char_list=special_char_list,
        is_predict=True, mode=mode)

    label_num = len(data_id2label)
    labels_need = list(data_id2label.keys())
    O_index = labels_need.index(data_label2id['O'])
    labels_need.pop(O_index)
    PAD_index = labels_need.index(data_label2id['PAD'])
    labels_need.pop(PAD_index)
    if model_type.lower().split('_')[1][-5:] == 'large':
        hidden_size = 1024
    else:
        hidden_size = 768

    if model_type.lower().split('_')[-1] == 'liner':
        model = Trig_BertLiner_Model(MODEL_PATH,tokenizer,label_num,
                hidden_size=hidden_size, label2id = data_label2id)
    elif model_type.lower().split('_')[-1] == 'crf':
        label_num = len(data_id2label) - 1
        model = Trig_BertCRF_Model(MODEL_PATH, tokenizer, label_num,
                hidden_size=hidden_size,device= device)
    elif model_type.lower().split('_')[-1] == 'multatt':
        model = Trig_BertMultAtt_Model(MODEL_PATH, tokenizer, label_num,
                head_num=12, att_output_dim = 96,device= device)
    else:
        raise NotImplementedError("没有定义模型")

    model.to(device)


    model.load_state_dict(torch.load(dee_model_name))
    model.predict_part(model, test_loader,
                            data_id2label,pred_sents_file,predict_save_path, device)


def do_arg_train():
    pass

def do_arg_predict():
    pass

if __name__ == '__main__':

    # if args.n_gpu > 1 and args.do_train:
    #     pass
        #to do
    pred_sents_file = data_dir + 'test.json'

    if do_train_mark or do_predict_mark:
        trig_dict_file = config_dir + 'trigger_tag.dict'
        trig_data_dir = data_dir + 'trigger/'
        trig_train_data_file = trig_data_dir + 'train.tsv'
        trig_dev_data_file = trig_data_dir + 'dev.tsv'
        trig_test_data_file = trig_data_dir + 'test.tsv'
        dee_trig_model_name = model_save_dir + model_type + '_see_trig_model.weight'
        trig_dict_save_file = model_save_dir + 'trig_dict.dict'
        trig_predict_save_path = model_save_dir + model_type + '_trig_test_pred.json'

    if do_arg_train_mark or do_arg_predict_mark:
        arg_dict_file = config_dir + 'role_tag.dict'
        arg_data_dir = data_dir + 'role/'
        arg_train_data_file = arg_data_dir + 'train.tsv'
        arg_dev_data_file = arg_data_dir + 'dev.tsv'
        arg_test_data_file = arg_data_dir + 'test.tsv'
        dee_arg_model_name = model_save_dir + model_type + '_see_arg_model.weight'
        arg_dict_save_file = model_save_dir + 'arg_dict.dict'
        arg_predict_save_path = model_save_dir + model_type + '_arg_test_pred.json'

    mode = 'trig'
    if do_train_mark:
        do_trig_train(trig_train_data_file,trig_dev_data_file,trig_test_data_file,
                      trig_dict_file,trig_dict_save_file,dee_trig_model_name,model_type,mode)
    elif do_predict_mark:
        do_trig_predict(trig_test_data_file,pred_sents_file, trig_dict_file,
            trig_dict_save_file, dee_trig_model_name, trig_predict_save_path, mode)

    mode = 'arg'
    if do_arg_train_mark:
        do_trig_train(arg_train_data_file, arg_dev_data_file,arg_test_data_file,
                      arg_dict_file,arg_dict_save_file, dee_arg_model_name, model_type, mode)
    elif do_arg_predict_mark:
        do_trig_predict(arg_test_data_file,pred_sents_file, arg_dict_file,
                        arg_dict_save_file, dee_arg_model_name, arg_predict_save_path, mode)












































