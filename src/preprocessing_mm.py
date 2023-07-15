import copy
from tqdm import tqdm
import os
import pandas as pd
import json


def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
        # for _ in range(100000):
            yield line
            line = f.readline()
    return

def conv(x):
    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])


def read_medmentions(raw_path, medmention_part = 'trng'):
    with open(os.path.join(raw_path, 'full/data/corpus_pubtator_pmids_'+medmention_part+'.txt'), 'r') as f:
        train_ids = f.readlines()
    for i in range(len(train_ids)):
        train_ids[i] = train_ids[i].strip('\n')
    
    with open(os.path.join(raw_path, 'st21pv/data/corpus_pubtator.txt'), 'r') as f:
        all_data = f.readlines()
    
    dataset = []
    buffer = dict()
    for i in tqdm(range(len(all_data))): 
        if '|t|' in all_data[i]:
            buffer['text'] = all_data[i].strip('\n').split('|', maxsplit=2)[-1]
            buffer['id'] = all_data[i].strip('\n').split('|', maxsplit=2)[0]
        elif '|a|' in all_data[i]:
            buffer['text'] += ' '
            buffer['text'] += all_data[i].strip('\n').split('|', maxsplit=2)[-1]
            buffer['annotations'] = list()
        elif buffer['id'] in all_data[i]:
            annotation = all_data[i].strip('\n').split('\t')
            buffer['annotations'].append({
                        'span_idx':(int(annotation[1]), int(annotation[2])), 
                        'mention':annotation[3], 
                        'semantic_type':annotation[4], 
                        'code_label':annotation[5].strip('UMLS:')})
        else:
            if buffer['id'] in train_ids:
                dataset.append(copy.deepcopy(buffer))

                
    return dataset


class UMLS_reader(object):

    def __init__(self, umls_path, source_range=None, lang_range=['ENG'], only_load_dict=False, debug=False):
        self.debug = debug
        self.umls_path = umls_path
        self.source_range = source_range
        self.lang_range = lang_range
        self.detect_type()

    def detect_type(self):
        if os.path.exists(os.path.join(self.umls_path, "MRCONSO.RRF")):
            self.type = "RRF"
        else:
            self.type = "txt"
    
    def generate_name_list_set(self, semantic_type, source_onto):
        name_reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        semantic_reader = byLineReader(os.path.join(self.umls_path, "MRSTY." + self.type))
        self.cui2pref = dict()
        self.cui_in_onto = set()
        for line in tqdm(name_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            lang = l[1]
            source = l[11]
            string = l[14]
            ispref = l[6]
            if lang == "ENG":
                if cui in self.cui2pref:
                    self.cui2pref[cui].append(string)
                else:
                    self.cui2pref[cui] = [string]
                if source in source_onto:
                    self.cui_in_onto.update([cui])
        self.cuis_in_semtc = {}
        for line in tqdm(semantic_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            semantic = l[1]
            type_str = l[3].lower()
            if semantic in semantic_type:
                self.cuis_in_semtc[cui] = type_str

        for cui in copy.deepcopy(list(self.cui2pref.keys())):
            if cui not in self.cuis_in_semtc or cui not in self.cui_in_onto:
                self.cui2pref.pop(cui)
        
        syn_count = 0
        for cui in self.cui2pref:
            self.cui2pref[cui] = list(set(conv(self.cui2pref[cui])))
            syn_count += len(self.cui2pref[cui])
        
        print("cui count:", len(self.cui2pref))
        print("synonyms count:", syn_count)


def prepare_umls_sty_subkb(stype):

    # semantic_type = set(['T005','T007','T017','T022','T031','T033','T037','T038','T058','T062','T074',
    #                 'T082','T091','T092','T097','T098','T103','T168','T170','T201','T204'])
    semantic_type = set([stype])
    semantic_type_ontology = pd.read_csv('./STY.csv')
    semantic_type_size = 0
    while len(semantic_type)!=semantic_type_size:
        semantic_type_size = len(semantic_type)
        for i in range(len(semantic_type_ontology)):
            if semantic_type_ontology['Parents'][i][-4:] in semantic_type:
                semantic_type.update([semantic_type_ontology['Class ID'][i][-4:]])
    source_onto = ['CPT','FMA','GO','HGNC','HPO','ICD10','ICD10CM','ICD9CM','MDR','MSH','MTH',
                    'NCBI','NCI','NDDF','NDFRT','OMIM','RXNORM','SNOMEDCT_US']
    UMLS = UMLS_reader('/media/sdb1/Hongyi_Yuan/UMLS2017AA', only_load_dict = True)
    UMLS.generate_name_list_set(semantic_type, source_onto)

    with open('./dataset/umls_subset_'+stype+'.json', 'w') as f:
        json.dump({k:[] for k in UMLS.cui2pref}, f)

def prepare_umls_snomed_subkb():

    semantic_type = set(['T005','T007','T017','T022','T031','T033','T037','T038','T058','T062','T074',
                    'T082','T091','T092','T097','T098','T103','T168','T170','T201','T204'])
    semantic_type_ontology = pd.read_csv('./STY.csv')
    semantic_type_size = 0
    while len(semantic_type)!=semantic_type_size:
        semantic_type_size = len(semantic_type)
        for i in range(len(semantic_type_ontology)):
            if semantic_type_ontology['Parents'][i][-4:] in semantic_type:
                semantic_type.update([semantic_type_ontology['Class ID'][i][-4:]])
    source_onto = ['SNOMEDCT_US']
    UMLS = UMLS_reader('/media/sdb1/Hongyi_Yuan/UMLS2017AA', only_load_dict = True)
    UMLS.generate_name_list_set(semantic_type, source_onto)

    snomed = {k:[] for k in UMLS.cui2pref}
    with open('./dataset/umls_subset_snomed.json', 'w') as f:
        json.dump(snomed, f)
    print(len(snomed))


if __name__ == '__main__':

    # prepare_umls_sty_subkb('T038')
    # prepare_umls_sty_subkb('T058')
    # prepare_umls_snomed_subkb()
    raw_path = '../MedMentions'

    for part in ['test', 'dev','trng']:
        dataset = read_medmentions(raw_path, part)
        json.dump(dataset, open(f"./dataset/medmention/{part}_text.json", "w"))
#        sty_count = {}
#        count_all = 0
#        for item in dataset:
#            for annot in item['annotations']:
#                if annot['semantic_type'] not in sty_count:
#                    sty_count[annot['semantic_type']] = 1
#                else:
#                    sty_count[annot['semantic_type']] += 1
#        print(sorted(sty_count.items(), key = lambda kv: kv[1], reverse = True))
#        input()
#

    


