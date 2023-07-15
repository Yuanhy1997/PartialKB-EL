import sys
sys.path.append("./GENRE")

from genre.fairseq_model import GENRE
from genre.trie import Trie
import pickle
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_fairseq as get_entity_spans
from genre.utils import get_entity_spans_post_processing, get_entity_spans_finalize, get_entity_spans_pre_processing, get_end_to_end_prefix_allowed_tokens_fn_fairseq
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import re

def ditch_error_sample_on_boundary(sample):

    count = 0
    for s in sample:
        if s == '{':
            count += 1
        elif s == '}':
            count -= 1
    if count == -1:
        sample = re.sub(r" \} \[[^\]\[]+\]", '', sample, count=1)
    elif count == 1:
        sample = sample[::-1]
        sample = sample.replace('{','', 1)
        sample = sample[::-1]

    return sample

def cal_metric_postprocessing(guess_entities, gold_entities, strs2cui, cui_list):
    gold_entities_code = []
    guess_entities_code = []

    bad_annotation = 0
    for item in gold_entities:
        if item[3].replace('_', ' ').strip(' ') in strs2cui and strs2cui[item[3].replace('_', ' ').strip(' ')] in cui_list:
            gold_entities_code.append((item[0], item[1], item[2], strs2cui[item[3].replace('_', ' ').strip(' ')]))
        else:
            bad_annotation += 1
            gold_entities_code.append((item[0], item[1], item[2], item[3].replace('_', ' ').strip(' ')))
    print('bad', bad_annotation)
    for item in guess_entities:
        if item[3].replace('_', ' ').strip(' ') in strs2cui:
            if strs2cui[item[3].replace('_', ' ').strip(' ')] in cui_list:
                guess_entities_code.append((item[0], item[1], item[2], strs2cui[item[3].replace('_', ' ').strip(' ')]))

    from genre.utils import (
        get_micro_precision,
        get_micro_recall,
        get_micro_f1,
    )
    guess_entities = guess_entities_code
    gold_entities = gold_entities_code

    micro_p = get_micro_precision(guess_entities, gold_entities)
    micro_r = get_micro_recall(guess_entities, gold_entities)
    micro_f1 = get_micro_f1(guess_entities, gold_entities)

    print(
    "OVERALL Micro P/R/F: {:.2f}/{:.2f}/{:.2f}".format(
        micro_p*100, micro_r*100, micro_f1*100
    )
    )

    ner_guess_entities = [item[:-1] for item in guess_entities]
    ner_gold_entities = [item[:-1] for item in gold_entities]
    micro_p = get_micro_precision(ner_guess_entities, ner_gold_entities)
    micro_r = get_micro_recall(ner_guess_entities, ner_gold_entities)
    micro_f1 = get_micro_f1(ner_guess_entities, ner_gold_entities)

    print(
    "NER Micro P/R/F: {:.2f}/{:.2f}/{:.2f}".format(
        micro_p*100, micro_r*100, micro_f1*100
    )
    )

    accurate_ner_guess_entities = []
    accurate_ner_gold_entities = []

    accurate_ners = []
    for item in ner_guess_entities:
        if item in ner_gold_entities:
            accurate_ners.append(item)

    for item in gold_entities:
        if item[:-1] in accurate_ners:
            accurate_ner_gold_entities.append(item)

    for item in guess_entities:
        if item[:-1] in accurate_ners:
            accurate_ner_guess_entities.append(item)

    micro_p = get_micro_precision(accurate_ner_guess_entities, accurate_ner_gold_entities)
    micro_r = get_micro_recall(accurate_ner_guess_entities, accurate_ner_gold_entities)
    micro_f1 = get_micro_f1(accurate_ner_guess_entities, accurate_ner_gold_entities)

    print(
    "ner correct samples, Micro P/R/F: {:.2f}/{:.2f}/{:.2f}".format(
        micro_p*100, micro_r*100, micro_f1*100
    )
    )


def cal_metric(guess_entities, gold_entities, strs2cui):
    gold_entities_code = []
    guess_entities_code = []

    bad_annotation = 0
    for item in gold_entities:
        if item[3].replace('_', ' ').strip(' ') in strs2cui:
            gold_entities_code.append((item[0], item[1], item[2], strs2cui[item[3].replace('_', ' ').strip(' ')]))
        else:
            bad_annotation += 1
            gold_entities_code.append((item[0], item[1], item[2], item[3].replace('_', ' ').strip(' ')))
    print('bad', bad_annotation)
    for item in guess_entities:
        if item[3].replace('_', ' ').strip(' ') in strs2cui:
            guess_entities_code.append((item[0], item[1], item[2], strs2cui[item[3].replace('_', ' ').strip(' ')]))

    from genre.utils import (
        get_micro_precision,
        get_micro_recall,
        get_micro_f1,
    )
    guess_entities = guess_entities_code
    gold_entities = gold_entities_code

    micro_p = get_micro_precision(guess_entities, gold_entities)
    micro_r = get_micro_recall(guess_entities, gold_entities)
    micro_f1 = get_micro_f1(guess_entities, gold_entities)

    print(
    "OVERALL Micro P/R/F: {:.2f}/{:.2f}/{:.2f}".format(
        micro_p*100, micro_r*100, micro_f1*100
    )
    )

    ner_guess_entities = [item[:-1] for item in guess_entities]
    ner_gold_entities = [item[:-1] for item in gold_entities]
    micro_p = get_micro_precision(ner_guess_entities, ner_gold_entities)
    micro_r = get_micro_recall(ner_guess_entities, ner_gold_entities)
    micro_f1 = get_micro_f1(ner_guess_entities, ner_gold_entities)

    print(
    "NER Micro P/R/F: {:.2f}/{:.2f}/{:.2f}".format(
        micro_p*100, micro_r*100, micro_f1*100
    )
    )

    accurate_ner_guess_entities = []
    accurate_ner_gold_entities = []

    accurate_ners = []
    for item in ner_guess_entities:
        if item in ner_gold_entities:
            accurate_ners.append(item)

    for item in gold_entities:
        if item[:-1] in accurate_ners:
            accurate_ner_gold_entities.append(item)

    for item in guess_entities:
        if item[:-1] in accurate_ners:
            accurate_ner_guess_entities.append(item)

    micro_p = get_micro_precision(accurate_ner_guess_entities, accurate_ner_gold_entities)
    micro_r = get_micro_recall(accurate_ner_guess_entities, accurate_ner_gold_entities)
    micro_f1 = get_micro_f1(accurate_ner_guess_entities, accurate_ner_gold_entities)

    print(
    "ner correct samples, Micro P/R/F: {:.2f}/{:.2f}/{:.2f}".format(
        micro_p*100, micro_r*100, micro_f1*100
    )
    )

def generate_output_scores(model, input_sentences, **kwargs):
    
    sentences = get_entity_spans_pre_processing(input_sentences)

    tokenized_sentences = [model.encode(sentence) for sentence in sentences]

    batched_hypos = model.generate(
        tokenized_sentences,
        **kwargs,
    )
    outputs = []
    for idx, hypos in enumerate(batched_hypos):
        if hypos:
            outputs.append({"text": hypos[0]["tokens"].cpu().numpy(), "positional_scores": hypos[0]["positional_scores"].cpu().numpy()})
        else:
            outputs.append({"text": tokenized_sentences[idx].cpu().numpy(), "positional_scores": -100*np.ones(len(tokenized_sentences[idx]))})
    # outputs = [ {"text": hypos[0]["tokens"].cpu().numpy(), "positional_scores": hypos[0]["positional_scores"].cpu().numpy()}
    #     for hypos in batched_hypos
    # ]
    return outputs

def thresholding_on_outputs(outputs, threshold, model):
    
    results = []
    for item in tqdm(outputs):
        if item['text'] == '':
            results.append('')
            continue
        ment_bos_idx = np.argwhere((item['text'] == 25522) + (item['text']==45152) > 0).squeeze(-1).tolist()
        ment_eos_idx = np.argwhere(item['text'] == 35524).squeeze(-1).tolist()
        all_eos_idx = np.argwhere(item['text'] == 27779).squeeze(-1).tolist()
        assert (len(ment_bos_idx) == len(ment_eos_idx)) & (len(all_eos_idx) == len(ment_eos_idx))
        item['text'] = item['text'].tolist()
        # print(model.decode(torch.tensor(item['text'])))
        if ment_bos_idx and threshold is not None:
            result_tok = item['text'][:ment_bos_idx[0]]
            for i in range(len(ment_bos_idx)):
                if np.mean(item["positional_scores"][ment_bos_idx[i]:all_eos_idx[i]]) > threshold:
                    result_tok += item['text'][ment_bos_idx[i]:all_eos_idx[i]+1]
                else:
                    result_tok += item['text'][ment_bos_idx[i]+1:ment_eos_idx[i]]
                print(np.mean(item["positional_scores"][ment_bos_idx[i]:all_eos_idx[i]]))
                
                if i == len(ment_bos_idx)-1:
                    result_tok += item['text'][all_eos_idx[i]+1:]
                else:
                    result_tok += item['text'][all_eos_idx[i]+1:ment_bos_idx[i+1]]
            results.append(list(result_tok))
        else:
            results.append(item['text'])
        # print(ment_bos_idx, all_eos_idx)
        # print(model.decode(torch.tensor(results[-1])))
        # input()
        item['text'] = np.array(item['text'])
        results[-1] = model.decode(torch.tensor(results[-1]))

    return results


if sys.argv[5] == 'bc5cdr':
    train_kb = sys.argv[1]
    test_kb = sys.argv[2]
    test_annot = sys.argv[3]

    if train_kb == 'all':        
        model_path = '/platform_tech/yuanzheng/GENE2E/saved_models/bc5cdr_refine_520/'
    elif train_kb == 'subset':
        model_path = '/platform_tech/yuanzheng/GENE2E/saved_models/bc5cdr_medicNmesh_524'
    elif train_kb == 'diffset':
        model_path = '/platform_tech/yuanzheng/GENE2E/saved_models/BC5CDR_mesh-medic'

    model = GENRE.from_pretrained(model_path, 
                                gpt2_encoder_json = os.path.join(model_path, 'encoder.json'), 
                                gpt2_vocab_bpe = os.path.join(model_path, 'vocab.bpe'),
                                max_len_a = 2).eval().cuda()
    # print(model.encode(' { } [ ] '))
    # print(model.encode('{ } [ ]'))
    print('model loaded')
    with open('./dataset/BC5CDR/raw_data/target_kb.json', 'r') as f:
        cui2strs = json.load(f)

    if test_kb == 'all':
        strs2cui = {}
        for cui in tqdm(cui2strs):
            for strs in cui2strs[cui]:
                strs2cui[strs] = cui
        trie_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_medicNmesh/trie.pkl'

    elif test_kb == 'subset':
        with open('./dataset/MEDIC_KB.json', 'r') as f:
            medic_cui2strs = json.load(f)
        strs2cui = {}
        for cui in tqdm(cui2strs):
            if cui in medic_cui2strs:
                for strs in cui2strs[cui]:
                    strs2cui[strs] = cui
        trie_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_medicNmesh/trie_in_medic.pkl'

    elif test_kb == 'diffset':
        with open('./dataset/MEDIC_KB.json', 'r') as f:
            medic_cui2strs = json.load(f)
        strs2cui = {}
        for cui in tqdm(cui2strs):
            if cui not in medic_cui2strs:
                for strs in cui2strs[cui]:
                    strs2cui[strs] = cui
        trie_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh-medic/trie_not_in_medic.pkl'


    if os.path.exists(trie_path):
        with open(trie_path, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
    else:
        trie = Trie(model.encode(" }} [ {} ]".format(e))[1:].tolist() for e in tqdm(strs2cui.keys()))
        with open(trie_path, "wb") as f:
            pickle.dump(trie.trie_dict, f)


    if test_annot == 'all':
        source_suffix = 'mesh'
    elif test_annot == 'subset':
        source_suffix = 'medicNmesh'
    elif test_annot == 'diffset':
        source_suffix = 'mesh-medic'

    eval_set = sys.argv[4]

    documents, documents_ref = {}, {}
    with open(f'/platform_tech/yuanzheng/GENE2E/BC5CDR_{source_suffix}/'+eval_set+'.source', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            documents[str(idx)] = line.strip('\n').replace(' } ', '')
    with open(f'/platform_tech/yuanzheng/GENE2E/BC5CDR_{source_suffix}/'+eval_set+'.target', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            documents_ref[str(idx)] = line.strip('\n')


    buffer_path = f'./train_on_{train_kb}_{eval_set}_on_{test_kb}_{sys.argv[6]}_of_{test_annot}.pkl'
    if os.path.exists(buffer_path):
        with open(buffer_path, 'rb') as f:
            output_sequences = pickle.load(f)

    else:
        input_list = []
        output_sequences = []
        for idx, sents in tqdm(enumerate(list(documents.values()))):
            input_list.append(sents)
            if len(input_list) == 8:
                outs = generate_output_scores(model, 
                                                input_list, 
                                                prefix_allowed_tokens_fn=get_end_to_end_prefix_allowed_tokens_fn_fairseq(
                                                                                model,
                                                                                get_entity_spans_pre_processing(input_list),
                                                                                candidates_trie=trie,
                                                                            )
                                                )
                output_sequences += outs
                input_list = []
        with open(buffer_path, 'wb') as f:
            pickle.dump(output_sequences, f)
    
    output_gold_sentences = get_entity_spans_post_processing(list(documents_ref.values()))
    gold_entities = get_entity_spans_finalize(list(documents.values()), output_gold_sentences)
    gold_entities = [(k,) + x for k, e in zip(documents.keys(), gold_entities) for x in e]

    if test_kb == 'all' and test_annot != 'all':
        print('post-processing....')
        if test_annot == 'subset':
            with open('./dataset/MEDIC_KB.json', 'r') as f:
                medic_cui2strs = json.load(f)
            cui_list = []
            for cui in tqdm(cui2strs):
                if cui in medic_cui2strs:
                    cui_list.append(cui)

        elif test_annot == 'diffset':
            with open('./dataset/MEDIC_KB.json', 'r') as f:
                medic_cui2strs = json.load(f)
            cui_list = []
            for cui in tqdm(cui2strs):
                if cui not in medic_cui2strs:
                    cui_list.append(cui)
        
        output_sequences_processed = thresholding_on_outputs(output_sequences, None, model)
        for idx, sent in enumerate(output_sequences_processed):
            if sent == '':
                output_sequences_processed[idx] = list(documents.values())[idx]
        output_guess_sentences = get_entity_spans_post_processing(output_sequences_processed)
        guess_entities = get_entity_spans_finalize(list(documents.values()), output_guess_sentences)
        guess_entities = [(k,) + x for k, e in zip(documents.keys(), guess_entities) for x in e]
        cal_metric_postprocessing(guess_entities, gold_entities, strs2cui, cui_list)

    else:


        for thres in range(0,10):
            thres *= -1/20
            output_sequences_processed = thresholding_on_outputs(output_sequences, thres, model)
            output_guess_sentences = get_entity_spans_post_processing(output_sequences_processed)
            guess_entities = get_entity_spans_finalize(list(documents.values()), output_guess_sentences)
            guess_entities = [(k,) + x for k, e in zip(documents.keys(), guess_entities) for x in e]
            cal_metric(guess_entities, gold_entities, strs2cui)

        output_sequences_processed = thresholding_on_outputs(output_sequences, None, model)
        output_guess_sentences = get_entity_spans_post_processing(output_sequences_processed)
        guess_entities = get_entity_spans_finalize(list(documents.values()), output_guess_sentences)
        guess_entities = [(k,) + x for k, e in zip(documents.keys(), guess_entities) for x in e]
        cal_metric(guess_entities, gold_entities, strs2cui)

elif sys.argv[5] == 'mm':
    
    train_kb = sys.argv[1]
    test_kb = sys.argv[2]
    test_annot = sys.argv[3]

    if train_kb == 'all':        
        model_path = '/platform_tech/yuanzheng/GENE2E/saved_models/medmentions_genre_all_kb/'
    elif train_kb == 'subset':
        model_path = f'/platform_tech/yuanzheng/GENE2E/saved_models/medmentions_genre_subset_kb_{sys.argv[6].lower()}/'
    elif train_kb == 'diffset':
        model_path = f'/platform_tech/yuanzheng/GENE2E/saved_models/medmentions_genre_diffset_{sys.argv[6].lower()}/'

    model = GENRE.from_pretrained(model_path, 
                                checkpoint_file='checkpoint_best.pt',
                                gpt2_encoder_json = os.path.join(model_path, 'encoder.json'), 
                                gpt2_vocab_bpe = os.path.join(model_path, 'vocab.bpe'),
                                max_len_a = 10,
                                no_early_stop = True).eval().cuda()
    # print(model.encode(' { } [ ] '))
    # print(model.encode('{ } [ ]'))
    print('model loaded')
    with open('/platform_tech/yuanzheng/GENE2E/MedMentions/st21pv_kb.json', 'r') as f:
        cui2strs = json.load(f)

    if test_kb == 'all':
        strs2cui = {}
        for cui in tqdm(cui2strs):
            for strs in cui2strs[cui]:
                strs2cui[strs] = cui
        trie_path = '/platform_tech/yuanzheng/GENE2E/MM_dataset/all_kb/trie.pkl'

    elif test_kb == 'subset':
        with open(f'./dataset/umls_subset_{sys.argv[6]}.json', 'r') as f:
            medic_cui2strs = json.load(f)
        strs2cui = {}
        for cui in tqdm(cui2strs):
            if cui in medic_cui2strs:
                for strs in cui2strs[cui]:
                    strs2cui[strs] = cui
        trie_path = f'/platform_tech/yuanzheng/GENE2E/MM_dataset/subset_kb_snomed/trie_in_{sys.argv[6]}.pkl'

    elif test_kb == 'diffset':
        with open(f'./dataset/umls_subset_{sys.argv[6]}.json', 'r') as f:
            medic_cui2strs = json.load(f)
        strs2cui = {}
        for cui in tqdm(cui2strs):
            if cui not in medic_cui2strs:
                for strs in cui2strs[cui]:
                    strs2cui[strs] = cui
        trie_path = f'/platform_tech/yuanzheng/GENE2E/MM_dataset/diffset_kb_snomed/trie_not_in_{sys.argv[6]}.pkl'

    print('number of cuis', len(set(strs2cui.values())))
    print('number of terms', len(set(strs2cui.keys())))


    if os.path.exists(trie_path):
        with open(trie_path, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
    else:
        trie = Trie(model.encode(" }} [ {} ]".format(e))[1:].tolist() for e in tqdm(strs2cui.keys()))
        with open(trie_path, "wb") as f:
            pickle.dump(trie.trie_dict, f)

    if test_annot == 'all':
        source_suffix = 'all_kb'
    elif test_annot == 'subset':
        source_suffix = f'subset_kb_{sys.argv[6]}'
    elif test_annot == 'diffset':
        source_suffix = f'diffset_kb_{sys.argv[6]}'

    eval_set = sys.argv[4]

    documents, documents_ref = {}, {}
    with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset/{source_suffix}/'+eval_set+'.source', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            documents[str(idx)] = line.strip('\n')
    with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset/{source_suffix}/'+eval_set+'.target', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            documents_ref[str(idx)] = ditch_error_sample_on_boundary(line.strip('\n'))


    buffer_path = f'/platform_tech/yuanzheng/GENE2E/MM_dataset/train_on_{train_kb}_{eval_set}_on_{test_kb}_{sys.argv[6]}_of_{test_annot}.pkl'
    if os.path.exists(buffer_path):
        with open(buffer_path, 'rb') as f:
            output_sequences = pickle.load(f)

    else:
        input_list = []
        output_sequences = []
        for idx, sents in tqdm(enumerate(list(documents.values()))):
            input_list.append(sents)
            if len(input_list) == 8:
                outs = generate_output_scores(model, 
                                                input_list, 
                                                prefix_allowed_tokens_fn=get_end_to_end_prefix_allowed_tokens_fn_fairseq(
                                                                                model,
                                                                                get_entity_spans_pre_processing(input_list),
                                                                                candidates_trie=trie,
                                                                            ),
                                                )
                output_sequences += outs
                input_list = []
        with open(buffer_path, 'wb') as f:
            pickle.dump(output_sequences, f)

    output_gold_sentences = get_entity_spans_post_processing(list(documents_ref.values()))
    gold_entities = get_entity_spans_finalize(list(documents.values()), output_gold_sentences)
    gold_entities = [(k,) + x for k, e in zip(documents.keys(), gold_entities) for x in e]

    if test_kb == 'all' and test_annot != 'all':
        print('post-processing....')
        if test_annot == 'subset':
            with open(f'./dataset/umls_subset_{sys.argv[6]}.json', 'r') as f:
                medic_cui2strs = json.load(f)
            cui_list = []
            for cui in tqdm(cui2strs):
                if cui in medic_cui2strs:
                    cui_list.append(cui)

        elif test_annot == 'diffset':
            with open(f'./dataset/umls_subset_{sys.argv[6]}.json', 'r') as f:
                medic_cui2strs = json.load(f)
            cui_list = []
            for cui in tqdm(cui2strs):
                if cui not in medic_cui2strs:
                    cui_list.append(cui)
        
        output_sequences_processed = thresholding_on_outputs(output_sequences, None, model)
        for idx, sent in enumerate(output_sequences_processed):
            if sent == '':
                output_sequences_processed[idx] = list(documents.values())[idx]
        output_guess_sentences = get_entity_spans_post_processing(output_sequences_processed)
        guess_entities = get_entity_spans_finalize(list(documents.values()), output_guess_sentences)
        guess_entities = [(k,) + x for k, e in zip(documents.keys(), guess_entities) for x in e]
        cal_metric_postprocessing(guess_entities, gold_entities, strs2cui, cui_list)

    else:


        for thres in range(0,10):
            thres *= -1/20
            print('=====Threshold:', thres)
            output_sequences_processed = thresholding_on_outputs(output_sequences, thres, model)
            for idx, sent in enumerate(output_sequences_processed):
                if sent == '':
                    output_sequences_processed[idx] = list(documents.values())[idx]
            output_guess_sentences = get_entity_spans_post_processing(output_sequences_processed)
            guess_entities = get_entity_spans_finalize(list(documents.values()), output_guess_sentences)
            guess_entities = [(k,) + x for k, e in zip(documents.keys(), guess_entities) for x in e]
            cal_metric(guess_entities, gold_entities, strs2cui)

        print('======No Thresholds=======')
        output_sequences_processed = thresholding_on_outputs(output_sequences, None, model)
        for idx, sent in enumerate(output_sequences_processed):
            if sent == '':
                output_sequences_processed[idx] = list(documents.values())[idx]
        output_guess_sentences = get_entity_spans_post_processing(output_sequences_processed)
        guess_entities = get_entity_spans_finalize(list(documents.values()), output_guess_sentences)
        guess_entities = [(k,) + x for k, e in zip(documents.keys(), guess_entities) for x in e]
        cal_metric(guess_entities, gold_entities, strs2cui)

else:

    train_kb = sys.argv[1]
    test_kb = sys.argv[2]
    test_annot = sys.argv[3]

    # if train_kb == 'all':        
    model_path = '/platform_tech/yuanzheng/GENE2E/saved_models/bc5cdr_refine_520/'

    model = GENRE.from_pretrained(model_path, 
                                gpt2_encoder_json = os.path.join(model_path, 'encoder.json'), 
                                gpt2_vocab_bpe = os.path.join(model_path, 'vocab.bpe'),
                                max_len_a = 2).eval().cuda()
    # print(model.encode(' { } [ ] '))
    # print(model.encode('{ } [ ]'))
    print('model loaded')
    with open('./dataset/BC5CDR/raw_data/target_kb.json', 'r') as f:
        cui2strs = json.load(f)

    
    strs2cui = {}
    for cui in tqdm(cui2strs):
        for strs in cui2strs[cui]:
            strs2cui[strs] = cui
    trie_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_medicNmesh/trie.pkl'

    if os.path.exists(trie_path):
        with open(trie_path, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
    else:
        trie = Trie(model.encode(" }} [ {} ]".format(e))[1:].tolist() for e in tqdm(strs2cui.keys()))
        with open(trie_path, "wb") as f:
            pickle.dump(trie.trie_dict, f)
    
    source_suffix = 'mesh'
    input_list = ['Indomethacin induced hypotension in sodium and volume depleted rats. After a single oral dose of 4 mg/kg indomethacin (IDM) to sodium and volume depleted rats plasma renin activity (PRA) and systolic blood pressure fell significantly within four hours .']

    output_sequences = []
    outs = generate_output_scores(model, 
                                input_list, 
                                prefix_allowed_tokens_fn=get_end_to_end_prefix_allowed_tokens_fn_fairseq(
                                                                model,
                                                                get_entity_spans_pre_processing(input_list),
                                                                candidates_trie=trie,
                                                            )
                                    )
    output_sequences += outs
    print(output_sequences)

    output_sequences_processed = thresholding_on_outputs(output_sequences, -100, model)
    print(output_sequences_processed)



    trie_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_medicNmesh/trie_in_medic.pkl'
    if os.path.exists(trie_path):
        with open(trie_path, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
    else:
        trie = Trie(model.encode(" }} [ {} ]".format(e))[1:].tolist() for e in tqdm(strs2cui.keys()))
        with open(trie_path, "wb") as f:
            pickle.dump(trie.trie_dict, f)
    
    source_suffix = 'mesh'
    input_list = ['Indomethacin induced hypotension in sodium and volume depleted rats. After a single oral dose of 4 mg/kg indomethacin (IDM) to sodium and volume depleted rats plasma renin activity (PRA) and systolic blood pressure fell significantly within four hours .']

    output_sequences = []
    outs = generate_output_scores(model, 
                                input_list, 
                                prefix_allowed_tokens_fn=get_end_to_end_prefix_allowed_tokens_fn_fairseq(
                                                                model,
                                                                get_entity_spans_pre_processing(input_list),
                                                                candidates_trie=trie,
                                                            )
                                    )
    output_sequences += outs
    print(output_sequences)

    output_sequences_processed = thresholding_on_outputs(output_sequences, -100, model)
    print(output_sequences_processed)





    # elif test_kb == 'subset':
    #     with open('./dataset/MEDIC_KB.json', 'r') as f:
    #         medic_cui2strs = json.load(f)
    #     strs2cui = {}
    #     for cui in tqdm(cui2strs):
    #         if cui in medic_cui2strs:
    #             for strs in cui2strs[cui]:
    #                 strs2cui[strs] = cui
    #     trie_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_medicNmesh/trie_in_medic.pkl'

    # elif test_kb == 'diffset':
    #     with open('./dataset/MEDIC_KB.json', 'r') as f:
    #         medic_cui2strs = json.load(f)
    #     strs2cui = {}
    #     for cui in tqdm(cui2strs):
    #         if cui not in medic_cui2strs:
    #             for strs in cui2strs[cui]:
    #                 strs2cui[strs] = cui
    #     trie_path = '/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh-medic/trie_not_in_medic.pkl'


    # if os.path.exists(trie_path):
    #     with open(trie_path, "rb") as f:
    #         trie = Trie.load_from_dict(pickle.load(f))
    # else:
    #     trie = Trie(model.encode(" }} [ {} ]".format(e))[1:].tolist() for e in tqdm(strs2cui.keys()))
    #     with open(trie_path, "wb") as f:
    #         pickle.dump(trie.trie_dict, f)


    # if test_annot == 'all':
    #     source_suffix = 'mesh'
    # elif test_annot == 'subset':
    #     source_suffix = 'medicNmesh'
    # elif test_annot == 'diffset':
    #     source_suffix = 'mesh-medic'

    # eval_set = sys.argv[4]

    # documents, documents_ref = {}, {}
    # with open(f'/platform_tech/yuanzheng/GENE2E/BC5CDR_{source_suffix}/'+eval_set+'.source', 'r') as f:
    #     for idx, line in enumerate(f.readlines()):
    #         documents[str(idx)] = line.strip('\n').replace(' } ', '')
    # with open(f'/platform_tech/yuanzheng/GENE2E/BC5CDR_{source_suffix}/'+eval_set+'.target', 'r') as f:
    #     for idx, line in enumerate(f.readlines()):
    #         documents_ref[str(idx)] = line.strip('\n')


    # buffer_path = f'./train_on_{train_kb}_{eval_set}_on_{test_kb}_{sys.argv[6]}_of_{test_annot}.pkl'
    # if os.path.exists(buffer_path):
    #     with open(buffer_path, 'rb') as f:
    #         output_sequences = pickle.load(f)

    # else:
    #     input_list = []
    #     output_sequences = []
    #     for idx, sents in tqdm(enumerate(list(documents.values()))):
    #         input_list.append(sents)
    #         if len(input_list) == 8:
    #             outs = generate_output_scores(model, 
    #                                             input_list, 
    #                                             prefix_allowed_tokens_fn=get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    #                                                                             model,
    #                                                                             get_entity_spans_pre_processing(input_list),
    #                                                                             candidates_trie=trie,
    #                                                                         )
    #                                             )
    #             output_sequences += outs
    #             input_list = []
    #     with open(buffer_path, 'wb') as f:
    #         pickle.dump(output_sequences, f)
    
    # output_gold_sentences = get_entity_spans_post_processing(list(documents_ref.values()))
    # gold_entities = get_entity_spans_finalize(list(documents.values()), output_gold_sentences)
    # gold_entities = [(k,) + x for k, e in zip(documents.keys(), gold_entities) for x in e]

    # if test_kb == 'all' and test_annot != 'all':
    #     print('post-processing....')
    #     if test_annot == 'subset':
    #         with open('./dataset/MEDIC_KB.json', 'r') as f:
    #             medic_cui2strs = json.load(f)
    #         cui_list = []
    #         for cui in tqdm(cui2strs):
    #             if cui in medic_cui2strs:
    #                 cui_list.append(cui)

    #     elif test_annot == 'diffset':
    #         with open('./dataset/MEDIC_KB.json', 'r') as f:
    #             medic_cui2strs = json.load(f)
    #         cui_list = []
    #         for cui in tqdm(cui2strs):
    #             if cui not in medic_cui2strs:
    #                 cui_list.append(cui)
        
    #     output_sequences_processed = thresholding_on_outputs(output_sequences, None, model)
    #     for idx, sent in enumerate(output_sequences_processed):
    #         if sent == '':
    #             output_sequences_processed[idx] = list(documents.values())[idx]
    #     output_guess_sentences = get_entity_spans_post_processing(output_sequences_processed)
    #     guess_entities = get_entity_spans_finalize(list(documents.values()), output_guess_sentences)
    #     guess_entities = [(k,) + x for k, e in zip(documents.keys(), guess_entities) for x in e]
    #     cal_metric_postprocessing(guess_entities, gold_entities, strs2cui, cui_list)

# CUDA_VISIBLE_DEVICES=2 python genre_eval.py all subset subset test case medic


# CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all diffset test mm T038
# CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all diffset test mm T038
# CUDA_VISIBLE_DEVICES=0 python genre_eval.py all diffset diffset test mm T038
# CUDA_VISIBLE_DEVICES=2 python genre_eval.py all subset subset test mm T038
# CUDA_VISIBLE_DEVICES=5 python genre_eval.py all all all test mm 
# CUDA_VISIBLE_DEVICES=6 python genre_eval.py diffset diffset diffset test mm T058
# CUDA_VISIBLE_DEVICES=7 python genre_eval.py subset subset subset test mm T058

# CUDA_VISIBLE_DEVICES=3 python genre_eval.py all subset subset test
# CUDA_VISIBLE_DEVICES=2 python genre_eval.py subset subset subset test mm T038
# CUDA_VISIBLE_DEVICES=6 python genre_eval.py subset subset subset dev

# for thres in range(0,10):
#     thres *= -1/20
#     output_sequences_processed = thresholding_on_outputs(output_sequences, thres, model)
#     output_guess_sentences = get_entity_spans_post_processing(output_sequences_processed)
#     guess_entities = get_entity_spans_finalize(list(documents.values()), output_guess_sentences)
#     guess_entities = [(k,) + x for k, e in zip(documents.keys(), guess_entities) for x in e]
#     print(f'threshold{thres}')
#     cal_metric(guess_entities, gold_entities, strs2cui)
#     input()


## old code
# input_list = []
# all_guess_entities = []
# for idx, sents in tqdm(enumerate(list(documents.values()))):
#     input_list.append(sents)
#     if len(input_list) == 8:
#         guess_entities = get_entity_spans(model, input_list, candidates_trie=trie, data_split = sys.argv[1])
#         all_guess_entities += guess_entities
#         input_list = []
# if input_list:
#     guess_entities = get_entity_spans(model, input_list, candidates_trie=trie, data_split = sys.argv[1])
#     all_guess_entities += guess_entities
#     input_list = []
# guess_entities = [(k,) + x for k, e in zip(documents.keys(), all_guess_entities) for x in e]


# output_gold_sentences = get_entity_spans_post_processing(list(documents_ref.values()))
# gold_entities = get_entity_spans_finalize(list(documents.values()), output_gold_sentences)
# gold_entities = [(k,) + x for k, e in zip(documents.keys(), gold_entities) for x in e]

# with open('./results_'+sys.argv[1]+'.json', 'w') as f:
#     json.dump([gold_entities, guess_entities], f, indent=2)

# # with open('./results_'+sys.argv[1]+'.json', 'r') as f:
# #     gold_entities, guess_entities = json.load(f)

