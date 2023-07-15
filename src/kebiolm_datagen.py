from preprocessing_bc5cdr import read_bc5cdr
from preprocessing_mm import read_medmentions
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import json
import csv
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
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh_kebiolm/'
    elif sys.argv[1] == 'subset':
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_medicNmesh_kebiolm/'
    elif sys.argv[1] == 'diffset':
        output_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh-medic_kebiolm/'

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

    elif sys.argv[1] == 'all':
        used_code2str = {}
        for code in code2str:
            if code2str[code]:
                used_code2str[code] = sorted(code2str[code], key = lambda x: len(x), reverse=False)[0]
        code2str = used_code2str

    else:
        RuntimeError

    for part in ['test', 'dev','train']:
        dataset = read_bc5cdr(path, part+'_corpus')
        source = []
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
                splitted_docs.append('{'+mention.replace('.', '')+'||||'+code+'{')
                bos_idx = sample["span_idx"][1]
            splitted_docs.append(output_text[bos_idx:])
            output_text = ''.join(splitted_docs)


            token_length= 0
            inps = ''
            for sent in sent_tokenize(output_text):
                token_length += len(sent.split(' '))
                inps += ' ' + sent
                if token_length > 150:
                    source.append(inps)
                    token_length= 0
                    inps = ''
            if inps:
                source.append(inps)
                token_length= 0
                inps = ''

        f = open(os.path.join(output_path, f'{part}.tsv'), 'w')
        g = open(os.path.join(output_path, f'{part}.code'), 'w')
        tsv_writer = csv.writer(f, delimiter = '\t')

        for sample in source:
            sample_pieces = sample.split('{')
            inp = []
            lab = []
            codes = []
            for piece in sample_pieces:
                if '||||' in piece:
                    mention, code = piece.split('||||')
                    toks = word_tokenize(mention)
                    inp += toks
                    for i in range(len(toks)):
                        lab.append('B' if i == 0 else 'I')
                    codes.append(code)
                else:
                    toks = word_tokenize(piece)
                    inp += toks
                    lab += ['O'] * len(toks)
                
            g.write(json.dumps(codes) + '\n')
            for t, l in zip(inp, lab):
                tsv_writer.writerow([t, l])
            tsv_writer.writerow([])
        f.close()
        
elif sys.argv[2] == 'mm':
    subkb = sys.argv[3]
    if sys.argv[1] == 'all':
        output_path = '/platform_tech/yuanzheng/GENE2E/MM_dataset/all_kb_kebiolm/'
    elif sys.argv[1] == 'subset':
        output_path = f'/platform_tech/yuanzheng/GENE2E/MM_dataset/subset_kb_kebiolm_{subkb}/'
    elif sys.argv[1] == 'diffset':
        output_path = f'/platform_tech/yuanzheng/GENE2E/MM_dataset/diffset_kb_kebiolm_{subkb}/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    path = '/platform_tech/yuanzheng/GENE2E/MedMentions'
    with open(os.path.join(path, 'st21pv_kb.json'), 'r') as f:
        code2str = json.load(f)
    with open(f'./dataset/umls_subset_{subkb}.json', 'r') as f:
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
                splitted_docs.append('{'+mention.replace('.', '')+'||||'+code+'{')
                bos_idx = sample["span_idx"][1]
            splitted_docs.append(output_text[bos_idx:])
            output_text = ''.join(splitted_docs)


            token_length= 0
            inps = ''
            for sent in sent_tokenize(output_text):
                token_length += len(sent.split(' '))
                inps += ' ' + sent
                if token_length > 80:
                    source.append(inps)
                    token_length= 0
                    inps = ''
            if inps:
                source.append(inps)
                token_length= 0
                inps = ''

        f = open(os.path.join(output_path, f'{part}.tsv'), 'w')
        g = open(os.path.join(output_path, f'{part}.code'), 'w')
        tsv_writer = csv.writer(f, delimiter = '\t')

        for sample in source:
            sample_pieces = sample.split('{')
            inp = []
            lab = []
            codes = []
            for piece in sample_pieces:
                if '||||' in piece:
                    mention, code = piece.split('||||')
                    toks = word_tokenize(mention)
                    inp += toks
                    for i in range(len(toks)):
                        lab.append('B' if i == 0 else 'I')
                    codes.append(code)
                else:
                    toks = word_tokenize(piece)
                    inp += toks
                    lab += ['O'] * len(toks)
                
            g.write(json.dumps(codes) + '\n')
            for t, l in zip(inp, lab):
                tsv_writer.writerow([t, l])
            tsv_writer.writerow([])
        f.close()
        
            
            
        




