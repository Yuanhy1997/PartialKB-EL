import json
import sys
import os
from preprocessing_mm import read_medmentions
from preprocessing_bc5cdr import read_bc5cdr

def get_entity_spans_pre_processing(sentences):
    if isinstance(sentences, list):

        return [
            (
                "{}".format(sent)
                .replace("\xa0", " ")
                .replace("{", "(")
                .replace("}", ")")
                .replace("[", "(")
                .replace("]", ")")
            )
            for sent in sentences
        ]
    
    else:

        return  ( "{}".format(sentences)
                .replace("\xa0", " ")
                .replace("{", "(")
                .replace("}", ")")
                .replace("[", "(")
                .replace("]", ")")
                .replace("(ABSTRACT TRUNCATED AT 250 WORDS)", '') )



for kb in ['all', 'subset', 'diffset']:
    print('='*100)
    path = './dataset/BC5CDR/raw_data/'
    with open(os.path.join(path, 'target_kb.json'), 'r') as f:
        code2str = json.load(f)
    with open('./dataset/MEDIC_KB.json', 'r') as f:
        medic_code2str = json.load(f)
    for code in list(code2str.keys()):
        if not code2str[code]:
            code2str.pop(code)

    if kb == 'all':
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh/'
    elif kb == 'subset':
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_medicNmesh/'
    elif kb == 'diffset':
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh-medic/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if kb == 'subset':
        used_code2str = {}
        for code in code2str:
            if code in medic_code2str and code2str[code]:
                used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
        code2str = used_code2str

    elif kb == 'diffset':
        used_code2str = {}
        for code in code2str:
            if code not in medic_code2str and code2str[code]:
                used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
        code2str = used_code2str

    elif kb != 'all':
        RuntimeError
    

    print(f'number of {kb} concepts in kb {len(code2str)}')

    concepts_train = []
    for part in ['train', 'test', 'dev']:
        dataset = read_bc5cdr(path, part+'_corpus')
        annotations = []
        concepts = []
        for article in dataset:
            for sample in article['annotations']:
                mention = get_entity_spans_pre_processing(sample['mention'])
                if '|' in sample['code_label']:
                    code = sample['code_label'].split('|')[0]
                else:
                    code = sample['code_label']
                if code not in code2str:
                    continue
                
                annotations.append((mention, code))
                concepts.append(code)

                if part == 'train':
                    concepts_train.append(code)
        
        print(f'number of {part} annotations {len(annotations)}')
        print(f'number of {part} concepts {len(set(concepts))}')
        if part != 'train':
            c = 0
            for item in annotations:
                if item[-1] in concepts_train:
                    c += 1
            print(f'number of {part} annotations {c} in train')
            print(f'number of {part} concepts n train {len(set(concepts).intersection(set(concepts_train)))}')






for subkb in ['t058', 'snomed']:
    print(subkb)

    for kb in ['all', 'diffset']:
        subkb_ = subkb.replace('t', 'T')
        with open(f'./dataset/umls_subset_{subkb_}.json', 'r') as f:
            medic_code2str = json.load(f)
        path = '/platform_tech/yuanzheng/GENE2E/MedMentions'
        with open(os.path.join(path, 'st21pv_kb.json'), 'r') as f:
            code2str = json.load(f)
        for code in list(code2str.keys()):
            if not code2str[code]:
                code2str.pop(code)

        if kb == 'all':
            output_path = '/platform_tech/yuanzheng/GENE2E/MM_dataset/all_kb'
        elif kb == 'subset':
            output_path = '/platform_tech/yuanzheng/GENE2E/MM_dataset/subset_kb_' + subkb
        elif kb == 'diffset':
            output_path = '/platform_tech/yuanzheng/GENE2E/MM_dataset/diffset_kb_' + subkb

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if kb == 'subset':
            used_code2str = {}
            for code in code2str:
                if code in medic_code2str and code2str[code]:
                    used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
            code2str = used_code2str

        elif kb == 'diffset':
            used_code2str = {}
            for code in code2str:
                if code not in medic_code2str and code2str[code]:
                    used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
            code2str = used_code2str

        elif kb != 'all':
            RuntimeError
        

        print(f'number of concepts in kb {len(code2str)}')

        concepts_train = []
        for part in ['train', 'test', 'dev']:
            dataset = read_medmentions(path,  part if part != 'train' else 'trng')
            annotations = []
            concepts = []
            for article in dataset:
                for sample in article['annotations']:
                    mention = get_entity_spans_pre_processing(sample['mention'])
                    if '|' in sample['code_label']:
                        code = sample['code_label'].split('|')[0]
                    else:
                        code = sample['code_label']
                    if code not in code2str:
                        continue
                    
                    annotations.append((mention, code))
                    concepts.append(code)

                    if part == 'train':
                        concepts_train.append(code)
            
            print(f'number of {part} annotations {len(annotations)}')
            print(f'number of {part} concepts {len(set(concepts))}')
            if part != 'train':
                c = 0
                for item in annotations:
                    if item[-1] not in concepts_train:
                        c += 1
                print(f'number of {part} annotations {c} not in train')
                print(f'number of {part} concepts not in train {len(set(concepts).difference(set(concepts_train)))}')

