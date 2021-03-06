import torch.nn as nn
from torch.nn.functional import softmax
import torch
from transformers import BertModel
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
import tqdm
import json
from module.utils import read_by_lines, write_by_lines, get_sentences

class Trig_BertLiner_Model(nn.Module):
    def __init__(self, MODEL_PATH, tokenizer, label_num, label2id={'O':0,'PAD':-100}):
        super(Trig_BertLiner_Model, self).__init__()
        model = BertModel.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = model
        self.tokenizer = tokenizer
        self.l_1 = nn.Linear(768, 384)
        self.l_2 = nn.Linear(384, label_num)
        # self.l_2 = nn.Linear(384, 192)
        # self.l_3 = nn.Linear(192, label_num)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.1)
        ignore_index = label2id['PAD']
        # ignore_index = -100
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask):

        out = self.model(input_ids, attention_mask)
        out = out[0]
        # out = out[0][:, -1, :]
        out = self.l_1(out)
        out = self.relu(out)
        out = self.l_2(out)
        # out = self.relu(out)
        # out = self.l_3(out)
        # out = self.softmax(out)
        return out

    def train_part(self,model, train_loader, optim, device):
        model.train()
        for batch in tqdm.tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask)
            output = torch.reshape(output, [output.shape[0] * output.shape[1], output.shape[2]])
            labels = torch.reshape(labels, [labels.shape[0] * labels.shape[1]])
            loss = self.loss_fn(output, labels)
            loss.backward()
            optim.step()

    def dev_part(self, model, dev_loader, labels_need, device):
        model.eval()
        # TP ????????????????????????????????????????????????
        # FN ????????????????????????????????????????????????
        # FP ????????????????????????????????????????????????
        # TF ????????????????????????????????????????????????
        # TP, TN, FN, FP = 0.00001, 0.00001, 0.00001, 0.00001
        # P, r, f1 = 0.00001, 0.00001, 0.00001
        # acc = 0.00001
        labels_true_all = np.array([])
        labels_pre_all = np.array([])
        for batch in tqdm.tqdm(dev_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_true = batch['labels'].cpu()
            output = model(input_ids, attention_mask)
            output = softmax(output, dim=-1).cpu()
            labels_pre = torch.argmax(output, dim=-1).cpu()
            labels_true = torch.reshape(labels_true, [-1])
            labels_pre = torch.reshape(labels_pre, [-1])
            labels_true_all = np.append(labels_true_all, labels_true)
            labels_pre_all = np.append(labels_pre_all, labels_pre)

        p = precision_score(labels_true_all, labels_pre_all, labels=labels_need, average='micro')
        r = recall_score(labels_true_all, labels_pre_all, labels=labels_need, average='micro')
        f1 = f1_score(labels_true_all, labels_pre_all, labels=labels_need, average='micro')
        acc = accuracy_score(labels_true_all, labels_pre_all)
        return p, r, f1, acc

    def predict_part(self, model, dev_loader, id2label, sentences_file,predict_save_path, device):
        model.eval()
        labels_true_all = np.array([])
        labels_pre_all = np.array([])
        results = []
        sentences = get_sentences(sentences_file)
        for batch in tqdm.tqdm(dev_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_true = batch['labels'].cpu()
            seq_lens = batch['senq_len'].cpu()
            output = model(input_ids, attention_mask)
            probs = softmax(output,dim=-1).cpu()
            # probs = probs.numpy()
            probs_ids = torch.argmax(output, dim=-1).cpu()
            # labels_true = torch.reshape(labels_true, [-1])
            # labels_pre = torch.reshape(labels_pre, [-1])
            # labels_true_all = np.append(labels_true_all, labels_true)
            # labels_pre_all = np.append(labels_pre_all, labels_pre)
            for p_list, p_ids, seq_len in zip(probs.tolist(), probs_ids.tolist(), seq_lens.tolist()):
                prob_one = [p_list[index][pid] for index, pid in enumerate(p_ids[1: seq_len - 1])]
                label_one = [id2label[pid] for pid in p_ids[1: seq_len - 1]]
                results.append({"probs": prob_one, "labels": label_one})
        assert len(results) == len(sentences)
        for sent, ret in zip(sentences, results):
            sent["pred"] = ret
        sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
        write_by_lines(predict_save_path, sentences)
        print("save data {} to {}".format(len(sentences), predict_save_path))
