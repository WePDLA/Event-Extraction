import os,json,copy
def get_formed_event_info(trigger, arguments_list, argument_tech, sentiment):
    '''
    将事件信息从字典格式转化为(trig1，arg_11, arg_12)\targument_tech格式
    '''
    formed_event_info = trigger
    for arg in arguments_list:  # 此时arguments为包含了同一种role的所有的argument
        # for arg in arguments:
        formed_event_info = formed_event_info + ', ' + arg
    formed_event_info = '(' + formed_event_info + ')'
    if len(argument_tech) == 0:
        argument_tech_info = 'NONE'
    else:
        argument_tech_info = argument_tech[0]
        for i in range(1, len(argument_tech)):
            argument_tech_info = argument_tech_info + '||' + argument_tech[i]
    formed_event_info = formed_event_info + '\t' + argument_tech_info + '\t' + sentiment
    return formed_event_info

def ann_read(ann_file, txt_file):
    '''
    input:
        ann_file: ann file path   ann文件中只能包含一个事件
        txt_file: txt file path
    '''
    with open(txt_file, 'r')as f1:
        txt_contect = f1.read().strip()
    with open(ann_file, 'r')as f2:
        ann_contect = f2.read().strip()
    ann_contect_list = ann_contect.split('\n')
    event_summary = ''
    event_info_list = []
    for line in ann_contect_list:
        if line[0] == 'E':
            event_summary = line
        else:
            event_info_list.append(line)
    assert event_summary != ''

    event_summary = event_summary.split('\t')[-1]
    trigger_summary = event_summary.split()[0]
    trigger_type = trigger_summary.split(':')[0]
    trigger_Tid = trigger_summary.split(':')[1]

    event_info_dic = {}
    trig_info_dic = {}
    arg_info_dic_lsit = []
    for line in event_info_list:
        if line[:len(trigger_Tid)] == trigger_Tid:
            trigger_info = line
            event_type = trigger_info.split('\t')[1].split()[0]
            trig_span_s = trigger_info.split('\t')[1].split()[1]
            trig_span_e = trigger_info.split('\t')[1].split()[2]
            trigger = trigger_info.split('\t')[-1]
            trig_info_dic['event_type'] = event_type
            trig_info_dic['trig_span_s'] = trig_span_s
            trig_info_dic['trig_span_e'] = trig_span_e
            trig_info_dic['trigger'] = trigger
        else:
            arg_info_dic = {}
            arg_role = line.split('\t')[1].split()[0]
            arg_span_s = line.split('\t')[1].split()[1]
            arg_span_e = line.split('\t')[1].split()[2]
            argument = line.split('\t')[-1]
            arg_info_dic['arg_role'] = arg_role
            arg_info_dic['arg_span_s'] = arg_span_s
            arg_info_dic['arg_span_e'] = arg_span_e
            arg_info_dic['argument'] = argument
            arg_info_dic_lsit.append(arg_info_dic)
    event_info_dic['trig_info'] = trig_info_dic
    event_info_dic['arg_info'] = arg_info_dic_lsit
    event_info_dic['text'] = txt_contect
    return event_info_dic

def ann2dict(ann_info, text_info = ''):
    '''
    input:
        ann_info: ann event
        text_info: sentence
    '''
    ann_contect_list = ann_info.split('\n')
    event_summary = ''
    event_info_list = []
    for line in ann_contect_list:
        if line[0] == 'E':
            event_summary = line
        else:
            event_info_list.append(line)
    assert event_summary != ''

    event_summary = event_summary.split('\t')[-1]
    trigger_summary = event_summary.split()[0]
    trigger_type = trigger_summary.split(':')[0]
    trigger_Tid = trigger_summary.split(':')[1]

    event_info_dic = {}
    trig_info_dic = {}
    arg_info_dic_lsit = []
    for line in event_info_list:
        if line[:len(trigger_Tid)] == trigger_Tid:
            trigger_info = line
            event_type = trigger_info.split('\t')[1].split()[0]
            trig_span_s = trigger_info.split('\t')[1].split()[1]
            trig_span_e = trigger_info.split('\t')[1].split()[2]
            trigger = trigger_info.split('\t')[-1]
            trig_info_dic['event_type'] = event_type
            trig_info_dic['trig_span_s'] = trig_span_s
            trig_info_dic['trig_span_e'] = trig_span_e
            trig_info_dic['trigger'] = trigger
        else:
            arg_info_dic = {}
            arg_role = line.split('\t')[1].split()[0]
            arg_span_s = line.split('\t')[1].split()[1]
            arg_span_e = line.split('\t')[1].split()[2]
            argument = line.split('\t')[-1]
            arg_info_dic['arg_role'] = arg_role
            arg_info_dic['arg_span_s'] = arg_span_s
            arg_info_dic['arg_span_e'] = arg_span_e
            arg_info_dic['argument'] = argument
            arg_info_dic_lsit.append(arg_info_dic)
    event_info_dic['trig_info'] = trig_info_dic
    event_info_dic['arg_info'] = arg_info_dic_lsit
    event_info_dic['text'] = text_info
    return event_info_dic

def get_files(file_dir):
    files = [files for root, dirs, files in os.walk(file_dir)]
    file_list = [file for file in files[-1] if file[-4:] == '.txt']
    return file_list
    # return files




























































