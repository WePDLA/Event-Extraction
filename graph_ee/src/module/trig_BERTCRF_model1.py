import torch.nn as nn
import torch
from transformers import BertModel
from module.CRF import CRF
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
from torch.nn.functional import softmax
import tqdm
import json
from module.utils import read_by_lines, write_by_lines, get_sentences

class Trig_BertCRF_Model(nn.Module):
    def __init__(self, MODEL_PATH, tokenizer, label_num, hidden_size = 768,
        device=torch.device("cuda"),label2id = None):
        super(Trig_BertCRF_Model, self).__init__()
        model = BertModel.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.label2id = label2id
        self.device = device
        self.l_1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
            nn.ReLU()
        )
        self.l_2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(int(self.hidden_size / 2), label_num + 2)
        )
        ## CRF - trigger
        ## trigger_size应为label_num
        kwargs = dict({'target_size': label_num, 'device': device})
        self.tri_CRF1 = CRF(**kwargs)

    def forward(self, input_ids, attention_mask):

        out = self.model(input_ids, attention_mask)
        out = out[0]
        out = self.l_1(out)
        l2_embedding = self.l_2(out)
        ## tri_CRF1 ##

        ## 去掉CLS和SEP字符


        _, result = self.tri_CRF1.forward(feats=l2_embedding,
                                        mask=attention_mask)
        return result, l2_embedding

    def loss_fn(self, feats, mask, tags):
        trigger_loss = self.tri_CRF1.neg_log_likelihood_loss(feats=feats,
                                                    mask=mask, tags=tags)
        return trigger_loss

    def train_part(self,model, train_loader, optim, device):
        model.train()
        for batch in tqdm.tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask']
            attention_mask = batch['attention_mask'].byte()
            attention_mask_loss = batch['attention_mask'].bool()
            attention_mask_loss = torch.BoolTensor(attention_mask_loss).to(device)
            attention_mask = torch.ByteTensor(attention_mask).to(device)
            labels = batch['labels'].to(device)
            output, embeddings = model(input_ids, attention_mask)
            loss = self.loss_fn(embeddings, attention_mask_loss, labels)
            loss.backward()
            optim.step()

    def accuracy_score(self, labels_true_all, labels_pre_all):
        num = len(labels_pre_all)
        equal = 0
        gold = 0
        for i in range(num):
            if labels_true_all[i] != self.label2id['PAD'] and labels_true_all[i] != self.label2id['O'] and labels_pre_all[i] != self.label2id['PAD'] and labels_pre_all[i] != self.label2id['O']:
                gold +=1

            if labels_pre_all[i] == labels_true_all[i] and labels_true_all[i] != self.label2id['PAD'] and labels_true_all[i] != self.label2id['O']:
                equal += 1
        if gold != 0:
            acc = equal / gold
        else:
            acc = 0
        return acc

    def dev_part(self, model, dev_loader, labels_need, device):
        model.eval()
        # TP 真实值是真，且模型认为是真的数量
        # FN 真实值是真，且模型认为是假的数量
        # FP 真实值是假，且模型认为是真的数量
        # TF 真实值是假，且模型认为是假的数量
        # TP, TN, FN, FP = 0.00001, 0.00001, 0.00001, 0.00001
        # P, r, f1 = 0.00001, 0.00001, 0.00001
        # acc = 0.00001
        labels_true_all = np.array([])
        labels_pre_all = np.array([])
        for batch in tqdm.tqdm(dev_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_true = batch['labels'].cpu()
            labels_pre, embeddings = model(input_ids, attention_mask)
            # labels_pre = torch.argmax(output, dim=-1).cpu()
            labels_pre = labels_pre.cpu()
            labels_true = torch.reshape(labels_true, [-1])
            labels_pre = torch.reshape(labels_pre, [-1])
            labels_true_all = np.append(labels_true_all, labels_true)
            labels_pre_all = np.append(labels_pre_all, labels_pre)

        p = precision_score(labels_true_all, labels_pre_all, labels=labels_need, average='micro')
        r = recall_score(labels_true_all, labels_pre_all, labels=labels_need, average='micro')
        f1 = f1_score(labels_true_all, labels_pre_all, labels=labels_need, average='micro')
        acc = self.accuracy_score(labels_true_all, labels_pre_all)
        return p, r, f1, acc

    def predict_part(self, model, test_loader, id2label, sentences_file,predict_save_path, device):
        model.eval()
        results = []
        sentences = get_sentences(sentences_file)
        for batch in tqdm.tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_lens = batch['senq_len'].cpu()
            probs_ids,_ = model(input_ids, attention_mask)
            for p_ids, seq_len in zip(probs_ids.tolist(), seq_lens.tolist()):
                prob_one = [0]
                label_one = [id2label[pid] for pid in p_ids[1: seq_len - 1]]
                results.append({"probs": prob_one, "labels": label_one})
        assert len(results) == len(sentences)
        for sent, ret in zip(sentences, results):
            sent["pred"] = ret
        sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
        write_by_lines(predict_save_path, sentences)
        print("save data {} to {}".format(len(sentences), predict_save_path))

