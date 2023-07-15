from preprocessing_bc5cdr import read_bc5cdr
from preprocessing_mm import read_medmentions
import os
from nltk.tokenize import sent_tokenize
import re
import json

import sys



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

if sys.argv[2] == 'bc5cdr':
    if sys.argv[1] == 'all':
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh/'
    elif sys.argv[1] == 'subset':
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_medicNmesh/'
    elif sys.argv[1] == 'diffset':
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh-medic/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    path = './dataset/BC5CDR/raw_data/'
    with open(os.path.join(path, 'target_kb.json'), 'r') as f:
        code2str = json.load(f)
    with open('./dataset/MEDIC_KB.json', 'r') as f:
        medic_code2str = json.load(f)

    if sys.argv[1] == 'subset':
        used_code2str = {}
        for code in code2str:
            if code in medic_code2str and code2str[code]:
                used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
        code2str = used_code2str

    elif sys.argv[1] == 'diffset':
        used_code2str = {}
        for code in code2str:
            if code not in medic_code2str and code2str[code]:
                used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
        code2str = used_code2str

    elif sys.argv[1] != 'all':
        RuntimeError

    for part in ['test', 'dev','train']:
        dataset = read_bc5cdr(path, part+'_corpus')
        source = []
        target = []
        for article in dataset:
            article['text'] = get_entity_spans_pre_processing(article['text'])
            input_text = article['text']
            output_text = article['text']

            splitted_docs = []
            bos_idx = 0
            for sample in article['annotations']:
                mention = get_entity_spans_pre_processing(sample['mention'])
                if '|' in sample['code_label']:
                    code = sample['code_label'].split('|')[0]
                else:
                    code = sample['code_label']
                if code not in code2str:
                    continue
                else:
                    concept = code2str[code]
                
                splitted_docs.append(output_text[bos_idx:sample["span_idx"][0]])
                splitted_docs.append('{ '+mention.replace('.', '')+' } [ '+concept+' ]')
                bos_idx = sample["span_idx"][1]
            splitted_docs.append(output_text[bos_idx:])
            output_text = ''.join(splitted_docs)


            token_length= 0
            inps, outs = '', ''
            for sent in sent_tokenize(output_text):
                token_length += len(sent.split(' '))
                outs += ' ' + sent
                inps += ' ' + re.sub(r"\[[^\]\[]+\]", '', sent).replace('{ ', '').replace(' } ', '')
                # if '[' in inps or ']' in inps:
                #     print(sent)
                #     print(re.sub(r"{[^{}]+}", '', sent))
                #     print(re.sub(r"{[^{}]+}", '', sent).replace('[ ', '').replace(' ] ', ''))
                #     input()
                if token_length > 150:
                    source.append(inps)
                    target.append(outs)
                    token_length= 0
                    inps, outs = '', ''
            if inps:
                source.append(inps)
                target.append(outs)
                token_length= 0
                inps, outs = '', ''

        with open(os.path.join(output_path, f'{part}.source'), 'w') as f, open(os.path.join(output_path, f'{part}.target'), 'w') as g:
            for s, t in zip(source, target):
                f.write(s+'\n')
                g.write(t+'\n')
        
elif sys.argv[2] == 'mm':
    subkb = sys.argv[3]
    if sys.argv[1] == 'all':
        output_path = '/platform_tech/yuanzheng/GENE2E/MM_dataset/all_kb'
    elif sys.argv[1] == 'subset':
        output_path = '/platform_tech/yuanzheng/GENE2E/MM_dataset/subset_kb_' + subkb
    elif sys.argv[1] == 'diffset':
        output_path = '/platform_tech/yuanzheng/GENE2E/MM_dataset/diffset_kb_' + subkb


    with open(f'./dataset/umls_subset_{subkb}.json', 'r') as f:
        medic_code2str = json.load(f)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    path = '/platform_tech/yuanzheng/GENE2E/MedMentions'
    with open(os.path.join(path, 'st21pv_kb.json'), 'r') as f:
        code2str = json.load(f)

    if sys.argv[1] == 'subset':
        used_code2str = {}
        for code in code2str:
            if code in medic_code2str and code2str[code]:
                used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
        code2str = used_code2str

    elif sys.argv[1] == 'diffset':
        used_code2str = {}
        for code in code2str:
            if code not in medic_code2str and code2str[code]:
                used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
        code2str = used_code2str

    elif sys.argv[1] == 'all':
        used_code2str = {}
        for code in code2str:
            if code2str[code]:
                used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
        code2str = used_code2str
    
    else:
        RuntimeError

    for part in ['test', 'dev','train']:
        dataset = read_medmentions(path, part if part != 'train' else 'trng')
        source = []
        target = []
        for article in dataset:
            article['text'] = get_entity_spans_pre_processing(article['text'])
            input_text = article['text']
            output_text = article['text']

            splitted_docs = []
            bos_idx = 0
            for sample in article['annotations']:
                mention = get_entity_spans_pre_processing(sample['mention'])
                if '|' in sample['code_label']:
                    code = sample['code_label'].split('|')[0]
                else:
                    code = sample['code_label']
                if code not in code2str:
                    continue
                else:
                    concept = code2str[code]
                
                splitted_docs.append(output_text[bos_idx:sample["span_idx"][0]])
                splitted_docs.append('{ '+mention.replace('.', '')+' } [ '+concept+' ]')
                bos_idx = sample["span_idx"][1]
            splitted_docs.append(output_text[bos_idx:])
            output_text = ''.join(splitted_docs)


            token_length= 0
            inps, outs = '', ''
            for sent in sent_tokenize(output_text):
                token_length += len(sent.split(' '))
                outs += ' ' + sent
                inps += ' ' + re.sub(r"\[[^\]\[]+\]", '', sent).replace('{ ', '').replace(' } ', '')
                # if '[' in inps or ']' in inps:
                #     print(sent)
                #     print(re.sub(r"{[^{}]+}", '', sent))
                #     print(re.sub(r"{[^{}]+}", '', sent).replace('[ ', '').replace(' ] ', ''))
                #     input()
                if token_length > 150:
                    source.append(inps)
                    target.append(outs)
                    token_length= 0
                    inps, outs = '', ''
            if inps:
                source.append(inps)
                target.append(outs)
                token_length= 0
                inps, outs = '', ''

        with open(os.path.join(output_path, f'{part}.source'), 'w') as f, open(os.path.join(output_path, f'{part}.target'), 'w') as g:
            for s, t in zip(source, target):
                f.write(s+'\n')
                g.write(t+'\n')
            
            




