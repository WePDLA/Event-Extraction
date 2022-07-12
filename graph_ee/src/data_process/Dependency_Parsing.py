import numpy as np
from fastHan import FastHan
from tqdm import  tqdm
model = FastHan(model_type='large')
max_len = 512
class DependencyParsing():
    # 对于中文，只将单词的第一个汉子之间做依存链接
    # 美国攻打伊拉克   （美 -> 攻 -> 伊）

    def __init__(self):
        self.relations_useful = {'sent':0,'nsubj':1,'nsubjpass':2,'dobj':3}

    def get_index_of_word(self, pars_res, pars_word_index):
        # 这个函数主要是用来计算字符在句子中的下标
        if pars_word_index == 0:
            return -1
        index = 0
        for i in range(pars_word_index - 1):
            index += len(pars_res[i][0])
        # +1是为了让'[CLS]'作为第0个标签
        return index + 1


    def change_index_to_char_level(self, pars_res):
        # 这个函数主要是将以单词为粒度的句法依存结果转换成以字符为粒度的结果
        # 注意，单词粒度的下标从1开始，字符粒度的下标也从1开始，但是root表示为-1，0为特殊字符'[CLS]'
        char_level_pars_res = []
        for token in pars_res:
            word = token[0]
            pars_word_index = token[1]
            pars_label = token[2]
            ner = token[3]
            word_start_index = self.get_index_of_word(pars_res, pars_word_index)
            for i in range(len(word)):
                char_level_pars_res.append([word[i], word_start_index, pars_label, ner])
        return char_level_pars_res


    # def get_pars_vec(self, sentences_list, max_seq_len = max_len):
    #     # 这个函数用来获得句法依存的稀疏向量表示
    #     # 这个版本的函数中，将所有的有用的relation都视为一种类型
    #     '''
    #     input: [sent_1,sent_2]
    #     output: [par_vec_1, par_vec_2], par_vec_1,shape() = len(sent_1) + 2
    #     +2 是考虑到要在句子中增加'[CLS]'和'[SEP]'两个特殊字符。
    #
    #     '''
    #     answer = []
    #     print("start dependency parsing ... ")
    #     for j in tqdm(range(len(sentences_list))):
    #         sentences_j = ''.join(sentences_list[j])
    #         answer_j = model([sentences_j], 'Parsing')
    #         answer.extend(answer_j)
    #     vec_list = []
    #     print("getting dp vec ... ")
    #     for i, sentence in enumerate(tqdm(answer)):
    #         vec_l = []
    #         pars_info = self.change_index_to_char_level(sentence)
    #         vec_size = len(pars_info) + 2
    #         # vec_size = max_seq_len
    #         if vec_size > max_seq_len:
    #             vec_size = max_seq_len
    #
    #         # for relation_i in self.relations_useful.keys():
    #         for relation_i in ['nsubj']:
    #             par_vec = np.zeros([vec_size, vec_size], dtype=bool)
    #             for char_j in range(len(pars_info)):
    #                 # 即这个char的依赖关系在我们定义的关系集合中
    #                 char_info = pars_info[char_j]
    #                 # if char_info[2] == relation_i:
    #                 if char_j >= max_seq_len - 2:
    #                     continue
    #                 if char_info[2] in self.relations_useful:
    #                     # +1是因为'[CLS]'是第0个标签
    #
    #                     char_index = char_j + 1
    #                     dep_char_index = char_info[1]
    #                     if dep_char_index >= max_seq_len-1 or char_index >= max_seq_len-2:
    #                         continue
    #                     par_vec[char_index][dep_char_index] = 1
    #                     par_vec[dep_char_index][char_index] = 1
    #             vec_l.append(par_vec)
    #         vec_list.append(vec_l)
    #     return vec_list



    # def get_pars_vec(self, sentences_list, max_seq_len = max_len):
    #     # 这个函数用来获得句法依存的稀疏向量表示
    #     # 这个版本的函数中，将所有的有用的relation都视为一种类型,并且将‘[cls]’字符作为sentence embedding
    #     # 为每一个字符和sent embedding 之间加上一条边
    #     '''
    #     input: [sent_1,sent_2]
    #     output: [par_vec_1, par_vec_2], par_vec_1,shape() = len(sent_1) + 2
    #     +2 是考虑到要在句子中增加'[CLS]'和'[SEP]'两个特殊字符。
    #
    #     '''
    #     answer = []
    #     print("start dependency parsing ... ")
    #     for j in tqdm(range(len(sentences_list))):
    #         sentences_j = ''.join(sentences_list[j])
    #         answer_j = model([sentences_j], 'Parsing')
    #         answer.extend(answer_j)
    #     vec_list = []
    #     print("getting dp vec ... ")
    #     for i, sentence in enumerate(tqdm(answer)):
    #         vec_l = []
    #         pars_info = self.change_index_to_char_level(sentence)
    #         len_pars_info = len(pars_info)
    #         vec_size = len(pars_info) + 2
    #         # vec_size = max_seq_len
    #         if vec_size > max_seq_len:
    #             vec_size = max_seq_len
    #
    #         # for relation_i in self.relations_useful.keys():
    #         for relation_i in ['nsubj']:
    #             par_vec = np.zeros([vec_size, vec_size], dtype=bool)
    #             for char_j in range(len(pars_info)):
    #                 # 即这个char的依赖关系在我们定义的关系集合中
    #                 char_info = pars_info[char_j]
    #                 # if char_info[2] == relation_i:
    #                 if char_j >= max_seq_len-2:
    #                     continue
    #                 par_vec[0][char_j + 1] = 1
    #                 par_vec[char_j + 1][0] = 1
    #                 if char_info[2] in self.relations_useful:
    #                     # +1是因为'[CLS]'是第0个标签
    #                     char_index = char_j + 1
    #                     dep_char_index = char_info[1]
    #                     if dep_char_index >= max_seq_len-1 or char_index >= max_seq_len-2:
    #                         continue
    #                     par_vec[char_index][dep_char_index] = 1
    #                     par_vec[dep_char_index][char_index] = 1
    #             vec_l.append(par_vec)
    #         vec_list.append(vec_l)
    #     return vec_list


    # def get_pars_vec(self, sentences_list, max_seq_len = max_len):
    #     # 这个函数用来获得句法依存的稀疏向量表示
    #     # 这个版本的函数中，将所有的有用的relation都视为不同类型,并且将‘[cls]’字符作为sentence embedding
    #     # 为每一个字符和sent embedding 之间加上一条边
    #     '''
    #     input: [sent_1,sent_2]
    #     output: [par_vec_1, par_vec_2], par_vec_1,shape() = len(sent_1) + 2
    #     +2 是考虑到要在句子中增加'[CLS]'和'[SEP]'两个特殊字符。
    #
    #     '''
    #     answer = []
    #     print("start dependency parsing ... ")
    #     for j in tqdm(range(len(sentences_list))):
    #         sentences_j = ''.join(sentences_list[j])
    #         answer_j = model([sentences_j], 'Parsing')
    #         answer.extend(answer_j)
    #     vec_list = []
    #     print("getting dp vec ... ")
    #     for i, sentence in enumerate(tqdm(answer)):
    #         vec_l = []
    #         pars_info = self.change_index_to_char_level(sentence)
    #         vec_size = len(pars_info) + 2
    #         # vec_size = max_seq_len
    #         if vec_size > max_seq_len:
    #             vec_size = max_seq_len
    #
    #         for relation_i in self.relations_useful.keys():
    #         # for relation_i in ['nsubj']:
    #             par_vec = np.zeros([vec_size, vec_size], dtype=bool)
    #             for char_j in range(len(pars_info)):
    #                 # 即这个char的依赖关系在我们定义的关系集合中
    #                 char_info = pars_info[char_j]
    #                 if char_info[2] == relation_i:
    #
    #                 # par_vec[0][char_j + 1] = 1
    #                 # par_vec[char_j + 1][0] = 1
    #                 # if char_info[2] in self.relations_useful:
    #                     # +1是因为'[CLS]'是第0个标签
    #
    #                     char_index = char_j + 1
    #                     dep_char_index = char_info[1]
    #                     if dep_char_index >= max_seq_len-1 or char_index >= max_seq_len-2:
    #                         continue
    #                     par_vec[char_index][dep_char_index] = 1
    #                     par_vec[dep_char_index][char_index] = 1
    #             vec_l.append(par_vec)
    #         vec_list.append(vec_l)
    #     return vec_list

    # def get_pars_vec(self, sentences_list, max_seq_len = max_len):
    #     # 这个函数用来获得句法依存的稀疏向量表示
    #     # 这个版本的函数中:
    #     # 1、将所有的有用的relation都视为一种类型
    #     # 3、为每一个单词加上自环
    #     # 4、这2种边在同一张图中
    #     '''
    #     input: [sent_1,sent_2]
    #     output: [par_vec_1, par_vec_2], par_vec_1,shape() = len(sent_1) + 2
    #     +2 是考虑到要在句子中增加'[CLS]'和'[SEP]'两个特殊字符。
    #
    #     '''
    #     answer = []
    #     print("start dependency parsing ... ")
    #     for j in tqdm(range(len(sentences_list))):
    #         sentences_j = ''.join(sentences_list[j])
    #         answer_j = model([sentences_j], 'Parsing')
    #         answer.extend(answer_j)
    #     vec_list = []
    #     print("getting dp vec ... ")
    #     for i, sentence in enumerate(tqdm(answer)):
    #         vec_l = []
    #         pars_info = self.change_index_to_char_level(sentence)
    #         len_pars_info = len(pars_info)
    #         vec_size = len(pars_info) + 2
    #         # vec_size = max_seq_len
    #         if vec_size > max_seq_len:
    #             vec_size = max_seq_len
    #
    #         # for relation_i in self.relations_useful.keys():
    #         for relation_i in ['loop']:
    #             par_vec = np.zeros([vec_size, vec_size], dtype=bool)
    #             for char_j in range(len(pars_info)):
    #                 # 即这个char的依赖关系在我们定义的关系集合中
    #                 char_info = pars_info[char_j]
    #                 # if char_info[2] == relation_i:
    #                 if char_j >= max_seq_len-2:
    #                     continue
    #
    #                 # if relation_i == 'sentemb':
    #                 # 为每一个字符加上该字符sentence embedding的边
    #                 # par_vec[0][char_j + 1] = 1
    #                 # par_vec[char_j + 1][0] = 1
    #
    #                 # 为一个单词加上自环
    #                 # if relation_i == 'loop':
    #                 par_vec[char_j + 1][char_j + 1] = 1
    #
    #                 if char_info[2] in self.relations_useful:
    #                     # +1是因为'[CLS]'是第0个标签
    #                     char_index = char_j + 1
    #                     dep_char_index = char_info[1]
    #                     if dep_char_index >= max_seq_len-1 or char_index >= max_seq_len-2:
    #                         continue
    #                     par_vec[char_index][dep_char_index] = 1
    #                     par_vec[dep_char_index][char_index] = 1
    #             vec_l.append(par_vec)
    #         vec_list.append(vec_l)
    #     return vec_list

    def get_pars(self,sentences_list, max_seq_len = max_len):
        # 这个含函数用来获得句法依存
        answer = []
        print("start dependency parsing ... ")
        for j in tqdm(range(len(sentences_list))):
            sentences_j = ''.join(sentences_list[j])
            answer_j = model([sentences_j], 'Parsing')
            answer.extend(answer_j)
        return answer


    def get_pars_vec(self, sentences_list, max_seq_len = max_len):
        # 这个函数用来获得句法依存的稀疏向量表示
        # 这个版本的函数中:
        # 1、将所有的有用的relation都视为一种类型,并且将‘[cls]’字符作为sentence embedding
        # 2、为每一个字符和sent embedding 之间加上一条边
        # 3、为每一个单词加上自环
        '''
        input: [sent_1,sent_2]
        output: [par_vec_1, par_vec_2], par_vec_1,shape() = len(sent_1) + 2
        +2 是考虑到要在句子中增加'[CLS]'和'[SEP]'两个特殊字符。

        '''
        answer = []
        print("start dependency parsing ... ")
        for j in tqdm(range(len(sentences_list))):
            sentences_j = ''.join(sentences_list[j])
            answer_j = model([sentences_j], 'Parsing')
            answer.extend(answer_j)
        vec_list = []
        print("getting dp vec ... ")
        for i, sentence in enumerate(tqdm(answer)):
            vec_l = []
            pars_info = self.change_index_to_char_level(sentence)
            len_pars_info = len(pars_info)
            vec_size = len(pars_info) + 2
            # vec_size = max_seq_len
            if vec_size > max_seq_len:
                vec_size = max_seq_len

            # for relation_i in self.relations_useful.keys():
            for relation_i in ['nsubj']:
                par_vec = np.zeros([vec_size, vec_size], dtype=bool)
                for char_j in range(len(pars_info)):
                    # 即这个char的依赖关系在我们定义的关系集合中
                    char_info = pars_info[char_j]
                    # if char_info[2] == relation_i:
                    if char_j >= max_seq_len-2:
                        continue
                    # 为每一个字符加上该字符sentence embedding的边
                    par_vec[0][char_j + 1] = 1
                    par_vec[char_j + 1][0] = 1

                    # 为一个单词加上自环
                    par_vec[char_j + 1][char_j + 1] = 1

                    if char_info[2] in self.relations_useful:
                        # +1是因为'[CLS]'是第0个标签
                        char_index = char_j + 1
                        dep_char_index = char_info[1]
                        if dep_char_index >= max_seq_len-1 or char_index >= max_seq_len-2:
                            continue
                        par_vec[char_index][dep_char_index] = 1
                        par_vec[dep_char_index][char_index] = 1
                vec_l.append(par_vec)
            vec_list.append(vec_l)
        return vec_list

    # def get_pars_vec(self, sentences_list, max_seq_len = max_len):
    #     # 这个函数用来获得句法依存的稀疏向量表示
    #     # 这个版本的函数中:
    #     # 1、将所有的有用的relation都视为一种类型,并且将‘[cls]’字符作为sentence embeddi ng
    #     # 2、为每一个字符和sent embedding 之间加上一条边
    #     # 3、为每一个单词加上自环
    #     # 4、这三种边分别在不同的图中
    #     '''
    #     input: [sent_1,sent_2]
    #     output: [par_vec_1, par_vec_2], par_vec_1,shape() = len(sent_1) + 2
    #     +2 是考虑到要在句子中增加'[CLS]'和'[SEP]'两个特殊字符。
    #
    #     '''
    #     answer = []
    #     print("start dependency parsing ... ")
    #     for j in tqdm(range(len(sentences_list))):
    #         sentences_j = ''.join(sentences_list[j])
    #         answer_j = model([sentences_j], 'Parsing')
    #         answer.extend(answer_j)
    #     vec_list = []
    #     print("getting dp vec ... ")
    #     for i, sentence in enumerate(tqdm(answer)):
    #         vec_l = []
    #         pars_info = self.change_index_to_char_level(sentence)
    #         len_pars_info = len(pars_info)
    #         vec_size = len(pars_info) + 2
    #         # vec_size = max_seq_len
    #         if vec_size > max_seq_len:
    #             vec_size = max_seq_len
    #
    #         # for relation_i in self.relations_useful.keys():
    #         for relation_i in ['sentemb','charedge','loop']:
    #             par_vec = np.zeros([vec_size, vec_size], dtype=bool)
    #             for char_j in range(len(pars_info)):
    #                 # 即这个char的依赖关系在我们定义的关系集合中
    #                 char_info = pars_info[char_j]
    #                 # if char_info[2] == relation_i:
    #                 if char_j >= max_seq_len-2:
    #                     continue
    #
    #                 if relation_i == 'sentemb':
    #                     # 为每一个字符加上该字符sentence embedding的边
    #                     par_vec[0][char_j + 1] = 1
    #                     par_vec[char_j + 1][0] = 1
    #
    #                 # 为一个单词加上自环
    #                 if relation_i == 'loop':
    #                     par_vec[char_j + 1][char_j + 1] = 1
    #
    #                 if relation_i == 'charedge' and (char_info[2] in self.relations_useful):
    #                     # +1是因为'[CLS]'是第0个标签
    #                     char_index = char_j + 1
    #                     dep_char_index = char_info[1]
    #                     if dep_char_index >= max_seq_len-1 or char_index >= max_seq_len-2:
    #                         continue
    #                     par_vec[char_index][dep_char_index] = 1
    #                     par_vec[dep_char_index][char_index] = 1
    #             vec_l.append(par_vec)
    #         vec_list.append(vec_l)
    #     return vec_list

if __name__ == "__main__":
    # sentences = ["最近，一位前便利蜂员工就因公司违规裁员，将便利蜂所在的公司虫极科技（北京）有限公司告上法庭。","阚志刚揍彭丽雯。","彭丽雯向阚志刚投降了。"]
    sentences = ["云南沃森生物技术股份有限公司关于股东解除股权质押的公告"]
    # sentences = ["12月，阚志刚喜欢彭丽雯。","彭丽雯被阚志刚揍了。"]
    for k in range(len(sentences)):
        sentences[k] = list(sentences[k])

    # sentences = ["12月，阚志刚喜欢彭丽雯。","彭丽雯被阚志刚揍了。"]


    pars_geter = DependencyParsing()
    pars_vecs = pars_geter.get_pars_vec(sentences)
    print('done')

