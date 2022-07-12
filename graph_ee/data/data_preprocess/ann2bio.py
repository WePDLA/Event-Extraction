import os, json
from data_process import ann_read, get_files

if __name__ == '__main__':
    source_data_dir = ""
    target_trig_dir = "./trigger_bio_data/"
    target_arg_dir = "./role_bio_data/"
    dir_list = [target_trig_dir, target_arg_dir]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.mkdir(dir)
    target_trig_file = target_trig_dir + 'trig_data_bio.tsv'
    target_arg_file = target_arg_dir + 'arg_data_bio.tsv'
    file_list = [target_trig_file, target_arg_file]
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)


    file_list = get_files(source_data_dir)
    for file in file_list:
        txt_file = source_data_dir + file
        ann_file = source_data_dir + file[:-4] + '.ann'
        event_info = ann_read(ann_file, txt_file)
        token_list = list(event_info['text'])
        event_type = event_info['trig_info']['event_type']
        trig_label_list = ['O' for i in range(len(token_list))]
        arg_label_list = ['O' for i in range(len(token_list))]
        token_s = int(event_info['trig_info']['trig_span_s'])
        token_e = int(event_info['trig_info']['trig_span_e'])
        trig_label_list[token_s] = 'B-{}'.format(event_type)
        if token_e - token_s > 1:
            for j in range(token_s + 1, token_e):
                trig_label_list[j] = 'I-{}'.format(event_type)
        for arg in event_info['arg_info']:
            role = arg['arg_role']
            arg_s = int(arg['arg_span_s'])
            arg_e = int(arg['arg_span_e'])
            arg_label_list[arg_s] = 'B-{}'.format(role)
            if arg_e - arg_s > 1:
                for k in range(arg_s + 1, arg_e):
                    arg_label_list[k] = 'I-{}'.format(role)

        with open(target_trig_file, 'a+')as f4:
            f4.write('\002'.join(token_list) + '\t' + '\002'.join(trig_label_list) + '\n')
        with open(target_arg_file, 'a+')as f5:
            f5.write('\002'.join(token_list) + '\t' + '\002'.join(arg_label_list) + '\n')
