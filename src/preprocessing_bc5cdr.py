import json
import copy
import os

def read_bc5cdr(path, data_split = 'train'):
    
    with open(os.path.join(path, data_split+'.txt'), 'r') as f:
        all_data = f.readlines()
        
    dataset = []
    buffer = dict()
    for i in range(len(all_data)): 
        if '|t|' in all_data[i]:
            buffer['text'] = all_data[i].strip('\n').split('|', maxsplit=2)[-1]
            buffer['id'] = all_data[i].strip('\n').split('|', maxsplit=2)[0]
        elif '|a|' in all_data[i]:
            buffer['text'] += ' '
            buffer['text'] += all_data[i].strip('\n').split('|', maxsplit=2)[-1]
            buffer['annotations'] = list()
        elif buffer['id'] in all_data[i]:
            annotation = all_data[i].strip('\n').split('\t')
            if len(annotation) == 6:
                buffer['annotations'].append({'span_idx':(int(annotation[1]), int(annotation[2])), 
                                              'mention':annotation[3], 
                                              'semantic_type':annotation[4], 
                                              'code_label':annotation[5]})
            elif len(annotation) > 6:
                if annotation[3] == ' '.join(annotation[6].split('|')):
                    bos_index = int(annotation[1])
                    for ment, cui in zip(annotation[6].split('|'), annotation[5].split('|')):
                        buffer['annotations'].append({'span_idx':(bos_index, bos_index+len(ment)), 
                                                      'mention':ment, 
                                                      'semantic_type':annotation[4], 
                                                      'code_label':cui})
                        assert buffer['text'][bos_index: bos_index+len(ment)] == ment
                        bos_index += len(ment) + 1
                else:
                    buffer['annotations'].append({'span_idx':(int(annotation[1]), int(annotation[2])), 
                                                  'mention':annotation[3], 
                                                  'semantic_type':annotation[4], 
                                                  'code_label':annotation[5]})
        else:
            dataset.append(copy.deepcopy(buffer))

    return dataset


if __name__ == '__main__':
    
    path = './dataset/BC5CDR/raw_data/'
    for part in ['test', 'dev','train']:
        dataset = read_bc5cdr(path, part+'_corpus')
        json.dump(dataset, open(path+part+".json", "w"))
