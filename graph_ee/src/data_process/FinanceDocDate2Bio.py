import os,json

"""
待测试的条目：
1、一篇文章中有两个同类型事件的情况   目测这种情况是有的
"""

def store_label_tag(args_dict, store_dir='../conf/fin_bio_data/'):
    with open(store_dir+'role_tag.dict', 'a+')as f1:
        for role in args_dict.keys():
            f1.write(str(args_dict[role]) + '\t' + role + '\n')



def store_event(sents_list, doc_labels_dict, bio_dir):
    """
    event_type:EquityFreeze  EquityRepurchase  EquityUnderweight  EquityOverweight  EquityPledge
    每种事件类型一个单独的文件。
    """
    sent_char_list = []
    for sent in sents_list:
        sent_char = '\002'.join(sent)
        sent_char_list.append(sent_char)
    doc_content = '\003'.join(sent_char_list)

    sent_test_list = []
    for sent in sents_list:
        sent_test_list.append(list(sent))

    for event_type in doc_labels_dict.keys():
        if event_type == 'EquityFreeze':
            bio_file = bio_dir + 'ef.tsv'
        elif event_type == 'EquityRepurchase':
            bio_file = bio_dir + 'er.tsv'
        elif event_type == 'EquityUnderweight':
            bio_file = bio_dir + 'eu.tsv'
        elif event_type == 'EquityOverweight':
            bio_file = bio_dir + 'eo.tsv'
        elif event_type == 'EquityPledge':
            bio_file = bio_dir + 'ep.tsv'

        label_list = []
        label_test_list = []
        sent_labels = doc_labels_dict[event_type]
        for sent_label in sent_labels:
            sent_label_continue = '\002'.join(sent_label)
            label_test_list.append(sent_label)
            label_list.append(sent_label_continue)
        label_content = '\003'.join(label_list)

        assert len(label_content.split('\002')) == len(doc_content.split('\002'))

        if not os.path.exists(bio_file):
            with open(bio_file, 'a+')as f1:
                f1.write('text_a\tlabel\n')

        with open(bio_file, 'a+')as f1:
            f1.write(doc_content + '\t' + label_content + '\n')


def trans_file(origin_file, bio_dir, is_store_label_tag = False):
    """
    event_type:EquityFreeze  EquityRepurchase  EquityUnderweight  EquityOverweight  EquityPledge
    每种事件类型一个单独的文件。
    """
    data_count = 0
    max_sent_num = 0
    type = bio_dir.split('/')[-2]
    label_tags = {}
    with open(origin_file, 'r')as f1:
        origin_event_list = json.load(f1)
    for origin_event in origin_event_list:
        data_count += 1
        id = origin_event[0]
        sents_list = origin_event[1]['sentences'] #文章是以多个句子的形式出现的
        if max_sent_num < len(sents_list):
            max_sent_num = len(sents_list)
        event_list = origin_event[1]['recguid_eventname_eventdict_list'] #这里面的内容是文章中出现的所有事件
        ann_mspan2dranges = origin_event[1]['ann_mspan2dranges'] # 这里存放了一些实体所在的句子，以及在句子中的span信息
        doc_labels_dict = {}
        for event in event_list:
            """
            目前对于同一个文档中存在的多个同类型的事件的处理方法是将它们的事件元素都放在同一组标签中
            """
            event_type = event[1]
            args_dict = event[2]
            if event_type not in doc_labels_dict.keys():
                labels_list = []
                for i in range(len(sents_list)):
                    labels_list.append(['O' for j in range(len(sents_list[i]))])
            else:
                labels_list = doc_labels_dict[event_type]

            for arg_role in args_dict.keys():
                """
                这里暂时默认每一个arg_role只有一个对应的arg
                """
                if args_dict[arg_role] is None:
                    pass
                else:
                    arg = args_dict[arg_role]
                    arg_mspan_list = ann_mspan2dranges[arg]

                    for arg_mspan in arg_mspan_list:
                        sent_idx = arg_mspan[0]
                        span_start = arg_mspan[1]
                        span_end = arg_mspan[2]
                        B_arg_role = 'B-' + arg_role
                        I_arg_role = 'I-' + arg_role
                        if B_arg_role not in label_tags.keys():
                            label_tags[B_arg_role] = len(label_tags)
                        if I_arg_role not in label_tags.keys():
                            label_tags[I_arg_role] = len(label_tags)
                        labels_list[sent_idx][span_start] = B_arg_role
                        for k in range(span_start + 1,span_end):
                            labels_list[sent_idx][k] = I_arg_role
            doc_labels_dict[event_type] = labels_list

        store_event(sents_list, doc_labels_dict, bio_dir)
        # print('done')
    if 'O' not in label_tags.keys():
        label_tags['O'] = len(label_tags)
    if is_store_label_tag:
        store_label_tag(label_tags)
    print("{} set {} data".format(type, data_count))
    print("max sent num is ", max_sent_num)

def mix_data(BioDocData_dir, BioDocData_mix_dir):
    type_list = ['train', 'dev', 'test']
    event_type_list = ['ef','er','eu','eo','ep']
    for type_i in type_list:
        aim_file = BioDocData_mix_dir + type_i + '.tsv'
        if os.path.exists(aim_file):
            os.remove(aim_file)
        for event_type_i in event_type_list:
            source_file = BioDocData_dir + type_i + '/' + event_type_i + '.tsv'
            with open(source_file, 'r')as f1:
                # 如果目标文件已经存在，则跳过第一行
                if os.path.exists(aim_file):
                    next(f1)
                source_content = f1.read()
            with open(aim_file, 'a+')as f2:
                f2.write(source_content)
            print('data ({}) done.'.format(source_file))


if __name__ == '__main__':

    FinanceDocDate_dir = '../../data/finance_doc_data/origin_data/'
    BioDocData_dir = '../../data/finance_doc_data/bio_data/'
    BioDocData_mix_dir = '../../data/finance_doc_data/bio_data_all/'


    # # FinanceDocDate_train_file = FinanceDocDate_dir + 'sample_train_1.json'
    # FinanceDocDate_train_file = FinanceDocDate_dir + 'train.json'
    # FinanceDocDate_dev_file = FinanceDocDate_dir + 'dev.json'
    # FinanceDocDate_test_file = FinanceDocDate_dir + 'test.json'
    # BioDocData_train_dir = BioDocData_dir + 'train/'
    # BioDocData_dev_dir = BioDocData_dir + 'dev/'
    # BioDocData_test_dir = BioDocData_dir + 'test/'
    #
    # trans_file(FinanceDocDate_train_file, BioDocData_train_dir, is_store_label_tag=True)
    # trans_file(FinanceDocDate_dev_file, BioDocData_dev_dir)
    # trans_file(FinanceDocDate_test_file, BioDocData_test_dir)

    mix_data(BioDocData_dir, BioDocData_mix_dir)



