

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json
import faiss
from tqdm.auto import tqdm 
import joblib
import pickle
import argparse

batch_size = 256

def get_bert_embed(phrase_list, m, tok, normalize=True, summary_method="CLS"):
    input_ids = []
    for phrase in tqdm(phrase_list):
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
    m.eval()
    count = len(input_ids)

    progressbar = tqdm(range(count // batch_size + 2))
    now_count = 0
    with torch.no_grad():
        while now_count < count:
            progressbar.update(1)
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(m.device)
            if summary_method == "CLS":
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            embed_np = embed.cpu().detach().numpy()
            if now_count == 0:
                output = embed_np
            else:
                output = np.concatenate((output, embed_np), axis=0)
            now_count = min(now_count + batch_size, count)

    return output

def main_overall(args, gold_spans):
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, config=config).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    with open('/platform_tech/yuanzheng/GENE2E/MedMentions/st21pv_kb.json', 'r') as f:
        code2strs = json.load(f)
    
    if args.test_kb == 'subset':
        with open(f'./dataset/umls_subset_{args.subkb}.json', 'r') as f:
            medic_code2strs = json.load(f)
        
        str2code = {}
        for code in code2strs:
            if code in medic_code2strs:
                for term in code2strs[code]:
                    if term in str2code:
                        str2code[term].append(code)
                    else:
                        str2code[term] = [code]
        term_list = list(str2code.keys())
        print('loading target kb term embeddings')

        if os.path.exists(f'/platform_tech/yuanzheng/mm_subset_coder_embed_{args.subkb}.pkl'):
            with open(f'/platform_tech/yuanzheng/mm_subset_coder_embed_{args.subkb}.pkl', 'rb') as f:
                embed_list = joblib.load(f)
        else:
            embed_list = get_bert_embed(term_list, model, tokenizer)
            with open(f'/platform_tech/yuanzheng/mm_subset_coder_embed_{args.subkb}.pkl', 'wb') as f:
                joblib.dump(embed_list, f)
   
    elif args.test_kb == 'all':

        str2code = {}
        for code in code2strs:
            for term in code2strs[code]:
                if term in str2code:
                    str2code[term].append(code)
                else:
                    str2code[term] = [code]
        term_list = list(str2code.keys())
        print('loading target kb term embeddings')
    
        if os.path.exists(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl'):
            with open(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl', 'rb') as f:
                embed_list = joblib.load(f)
        else:
            embed_list = get_bert_embed(term_list, model, tokenizer)
            with open(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl', 'wb') as f:
                joblib.dump(embed_list, f)
    
    elif args.test_kb == 'diffset':
        with open(f'./dataset/umls_subset_{args.subkb}.json', 'r') as f:
            medic_code2strs = json.load(f)
        
        str2code = {}
        for code in code2strs:
            if code not in medic_code2strs:
                for term in code2strs[code]:
                    if term in str2code:
                        str2code[term].append(code)
                    else:
                        str2code[term] = [code]
        term_list = list(str2code.keys())
        print('loading target kb term embeddings')
    
        if os.path.exists(f'/platform_tech/yuanzheng/mm_diffset_coder_embed_{args.subkb}.pkl'):
            with open(f'/platform_tech/yuanzheng/mm_diffset_coder_embed_{args.subkb}.pkl', 'rb') as f:
                embed_list = joblib.load(f)
        else:
            embed_list = get_bert_embed(term_list, model, tokenizer)
            with open(f'/platform_tech/yuanzheng/mm_diffset_coder_embed_{args.subkb}.pkl', 'wb') as f:
                joblib.dump(embed_list, f)
    
    if args.test_annot == 'all':
        annot = 'all_kb_kebiolm'
    elif args.test_annot == 'subset':
        annot = f'subset_kb_kebiolm_{args.subkb}'
    elif args.test_annot == 'diffset':
        annot = f'diffset_kb_kebiolm_{args.subkb}'
    # with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset/{annot}/{args.test_set}_gold_span.pkl', 'rb') as f:
    #     gold_spans = pickle.load(f)

    if args.train_kb == 'all':
        train_kb = 'all'
    elif args.train_kb == 'subset':
        train_kb = 'subset'
    elif args.train_kb == 'diffset':
        train_kb = 'diffset'
    with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset//{annot}/{train_kb}_train_result/{args.test_set}_predictions.txt', 'r') as f:
        ner_spans_pred = parse_ner_result([line.strip('\n') for line in f.readlines()])

    ner_mentions = []
    for pr in ner_spans_pred:
        ner_mentions.append(pr[1])
    print('loading ner term embeddings')
    mention_embed_list = get_bert_embed(ner_mentions, model, tokenizer)
    print(len(mention_embed_list))
    d = len(embed_list[0])
    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(embed_list.astype('float32'))

    k = 1           
    print('start faiss search')   
    D, I = quantizer.search(mention_embed_list.astype('float32'), k)

    overall_results = []
    for i, idx in enumerate(I):
        # try:
        overall_results.append(ner_spans_pred[i] + (str2code[term_list[int(idx[0])]][0],))
        # except:
        #     print(ner_spans_pred[i])
        #     print(idx[0])
        #     print(len(term_list))
        #     print(term_list[int(idx[0])])
        #     print(str2code[term_list[int(idx[0])]][0])
        #     input()
    
    precision = len(set(gold_spans).intersection(set(overall_results))) / len(overall_results)
    recall = len(set(gold_spans).intersection(set(overall_results))) / len(gold_spans)
    f1 = (2*precision*recall)/(precision+recall)
    print(f'final result(P/R/F1): {round(precision*100, 2)}/{round(recall*100, 2)}/{round(f1*100, 2)}')
    print(f'result count: {len(set(gold_spans).intersection(set(overall_results)))}, {len(overall_results)}')
    for threshold in range(30):
        overall_results = []
        for i, idx in enumerate(I):
            if np.sum(mention_embed_list[i]*embed_list[int(idx[0])]) > (threshold/30):
                overall_results.append(ner_spans_pred[i] + (str2code[term_list[int(idx[0])]][0],))
        
        precision = len(set(gold_spans).intersection(set(overall_results))) / len(overall_results)
        recall = len(set(gold_spans).intersection(set(overall_results))) / len(gold_spans)
        f1 = (2*precision*recall)/(precision+recall)
        print(f'threshold:{(threshold/30)}, OVERALL final result(P/R/F1): {round(precision*100, 2)}/{round(recall*100, 2)}/{round(f1*100, 2)}')
        # print(f'result count: {len(set(gold_spans).intersection(set(overall_results)))}, {len(overall_results)}')
        ner_gold = [item[:-1] for item in gold_spans]
        ner_pred = [item[:-1] for item in overall_results]
        precision = len(set(ner_gold).intersection(set(ner_pred))) / len(ner_pred)
        recall = len(set(ner_gold).intersection(set(ner_pred))) / len(ner_gold)
        f1 = (2*precision*recall)/(precision+recall)
        print(f'threshold:{(threshold/30)}, NER final result(P/R/F1): {round(precision*100, 2)}/{round(recall*100, 2)}/{round(f1*100, 2)}')
        print(f'result count: {len(set(gold_spans).intersection(set(overall_results)))}, {len(overall_results)}')


def main_postprocess(args, gold_spans):

    if not os.path.exists(f'/platform_tech/yuanzheng/GENE2E/kebiolm_all_results_on_{args.subkb}.pkl'):
        config = AutoConfig.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path, config=config).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        with open('/platform_tech/yuanzheng/GENE2E/MedMentions/st21pv_kb.json', 'r') as f:
            code2strs = json.load(f)

        str2code = {}
        for code in code2strs:
            for term in code2strs[code]:
                if term in str2code:
                    str2code[term].append(code)
                else:
                    str2code[term] = [code]
        term_list = list(str2code.keys())
        print('loading target kb term embeddings')
    
        if os.path.exists(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl'):
            with open(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl', 'rb') as f:
                embed_list = joblib.load(f)
        else:
            embed_list = get_bert_embed(term_list, model, tokenizer)
            with open(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl', 'wb') as f:
                joblib.dump(embed_list, f)
       
        if args.test_annot == 'all':
            annot = 'all_kb_kebiolm'
        elif args.test_annot == 'subset':
            annot = f'subset_kb_kebiolm_{args.subkb}'
        elif args.test_annot == 'diffset':
            annot = f'diffset_kb_kebiolm_{args.subkb}'
        train_kb = 'all'
        with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset//{annot}/{train_kb}_train_result/{args.test_set}_predictions.txt', 'r') as f:
            ner_spans_pred = parse_ner_result([line.strip('\n') for line in f.readlines()])

        ner_mentions = []
        for pr in ner_spans_pred:
            ner_mentions.append(pr[1])
        print('loading ner term embeddings')
        mention_embed_list = get_bert_embed(ner_mentions, model, tokenizer)
        print(len(mention_embed_list))
        d = len(embed_list[0])
        quantizer = faiss.IndexFlatL2(d)
        quantizer.add(embed_list.astype('float32'))

        k = 1           
        print('start faiss search')   
        D, I = quantizer.search(mention_embed_list.astype('float32'), k)

        overall_results = []
        for i, idx in enumerate(I):
            overall_results.append(ner_spans_pred[i] + (str2code[term_list[int(idx[0])]][0],))
        
        with open(f'/platform_tech/yuanzheng/GENE2E/kebiolm_all_results_on_{args.subkb}.pkl', 'wb') as f:
            pickle.dump(overall_results, f)
    else:
        with open(f'/platform_tech/yuanzheng/GENE2E/kebiolm_all_results_on_{args.subkb}.pkl', 'rb') as f:
            overall_results = pickle.load(f)
    
    all_results = list(overall_results)


    with open('/platform_tech/yuanzheng/GENE2E/MedMentions/st21pv_kb.json', 'r') as f:
            code2strs = json.load(f)
    with open(f'./dataset/umls_subset_{args.subkb}.json', 'r') as f:
            subset_code2strs = json.load(f)
    
    if args.test_annot == 'subset':
        cui_list = set(subset_code2strs.keys()).intersection(set(code2strs.keys()))
    elif args.test_annot == 'diffset':
        cui_list = set(code2strs.keys()).difference(set(subset_code2strs.keys()))
    else:
        RuntimeError
    
    overall_results = []
    for item in all_results:
        if item[-1] in cui_list:
            overall_results.append(item)

    precision = len(set(gold_spans).intersection(set(overall_results))) / len(overall_results)
    recall = len(set(gold_spans).intersection(set(overall_results))) / len(gold_spans)
    f1 = (2*precision*recall)/(precision+recall)
    print(f'final result(P/R/F1): {round(precision*100, 2)}/{round(recall*100, 2)}/{round(f1*100, 2)}')
    print(f'result count: {len(set(gold_spans).intersection(set(overall_results)))}, {len(overall_results)}')
    
    ner_gold = [item[:-1] for item in gold_spans]
    ner_pred = [item[:-1] for item in overall_results]
    precision = len(set(ner_gold).intersection(set(ner_pred))) / len(ner_pred)
    recall = len(set(ner_gold).intersection(set(ner_pred))) / len(ner_gold)
    f1 = (2*precision*recall)/(precision+recall)
    print(f'NER final result(P/R/F1): {round(precision*100, 2)}/{round(recall*100, 2)}/{round(f1*100, 2)}')
    print(f'result count: {len(set(gold_spans).intersection(set(overall_results)))}, {len(overall_results)}')


def main_correct_ner(args, gold_spans):
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, config=config).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    with open('/platform_tech/yuanzheng/GENE2E/MedMentions/st21pv_kb.json', 'r') as f:
        code2strs = json.load(f)

    if args.test_kb == 'subset':

        with open(f'./dataset/umls_subset_{args.subkb}.json', 'r') as f:
            medic_code2strs = json.load(f)
        
        str2code = {}
        for code in code2strs:
            if code in medic_code2strs:
                for term in code2strs[code]:
                    if term in str2code:
                        str2code[term].append(code)
                    else:
                        str2code[term] = [code]
        
        term_list = list(str2code.keys())
        print('loading target kb term embeddings')

        if os.path.exists(f'/platform_tech/yuanzheng/mm_subset_coder_embed_{args.subkb}.pkl'):
            with open(f'/platform_tech/yuanzheng/mm_subset_coder_embed_{args.subkb}.pkl', 'rb') as f:
                embed_list = joblib.load(f)
        else:
            embed_list = get_bert_embed(term_list, model, tokenizer)
            with open(f'/platform_tech/yuanzheng/mm_subset_coder_embed_{args.subkb}.pkl', 'wb') as f:
                joblib.dump(embed_list, f)
    elif args.test_kb == 'all':

        str2code = {}
        for code in code2strs:
            for term in code2strs[code]:
                if term in str2code:
                    str2code[term].append(code)
                else:
                    str2code[term] = [code]
        term_list = list(str2code.keys())
        print('loading target kb term embeddings')
        
        if os.path.exists(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl'):
            with open(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl', 'rb') as f:
                embed_list = joblib.load(f)
        else:
            embed_list = get_bert_embed(term_list, model, tokenizer)
            with open(f'/platform_tech/yuanzheng/mm_all_coder_embed.pkl', 'wb') as f:
                joblib.dump(embed_list, f)
    
    elif args.test_kb == 'diffset':

        with open(f'./dataset/umls_subset_{args.subkb}.json', 'r') as f:
            medic_code2strs = json.load(f)
        str2code = {}
        for code in code2strs:
            if code not in medic_code2strs:
                for term in code2strs[code]:
                    if term in str2code:
                        str2code[term].append(code)
                    else:
                        str2code[term] = [code]
        term_list = list(str2code.keys())
        print('loading target kb term embeddings')
    
    
        if os.path.exists(f'/platform_tech/yuanzheng/mm_diffset_coder_embed_{args.subkb}.pkl'):
            with open(f'/platform_tech/yuanzheng/mm_diffset_coder_embed_{args.subkb}.pkl', 'rb') as f:
                embed_list = joblib.load(f)
        else:
            embed_list = get_bert_embed(term_list, model, tokenizer)
            with open(f'/platform_tech/yuanzheng/mm_diffset_coder_embed_{args.subkb}.pkl', 'wb') as f:
                joblib.dump(embed_list, f)

    if args.test_annot == 'all':
        annot = 'all_kb_kebiolm'
    elif args.test_annot == 'subset':
        annot = f'subset_kb_kebiolm_{args.subkb}'
    elif args.test_annot == 'diffset':
        annot = f'diffset_kb_kebiolm_{args.subkb}'
    # with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset/{annot}/{args.test_set}_gold_span.pkl', 'rb') as f:
    #     gold_spans = pickle.load(f)

    if args.train_kb == 'all':
        train_kb = 'all'
    elif args.train_kb == 'subset':
        train_kb = 'subset'
    elif args.train_kb == 'diffset':
        train_kb = 'diffset'
    
        
    with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset/{annot}/{train_kb}_train_result/{args.test_set}_predictions.txt', 'r') as f:
        ner_spans_pred = parse_ner_result([line.strip('\n') for line in f.readlines()])
 
    ner_mentions = []
    ner_codes = []
    for go in gold_spans:
        if go[:-1] in ner_spans_pred:
            ner_mentions.append(go[1].lower())
            ner_codes.append(go[-1])
    print('loading ner term embeddings')
    mention_embed_list = get_bert_embed(ner_mentions, model, tokenizer)

    d = len(embed_list[0])
    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(embed_list.astype('float32'))

    k = 1           
    print('start faiss search')   
    D, I = quantizer.search(mention_embed_list.astype('float32'), k)
    code_results = []
    for idx in I:
        code_results.append(str2code[term_list[int(idx[0])]])
    
    assert len(ner_codes) == len(code_results)
    correct = 0
    for g, p in zip(ner_codes, code_results):
        if g in p:
            correct += 1
    print(f'final result {correct/len(ner_codes)}')

def parse_ner_result(inputs):
    golden_spans = []
    sample_idx = 0
    mention, bosidx, cur_bosidx = '', -1, 0
    for line in inputs:
        if not line:
            if bosidx >= 0:
                golden_spans.append((sample_idx, mention.strip(' '), bosidx, cur_bosidx-bosidx))
            sample_idx += 1
            mention, bosidx, cur_bosidx = '', -1, 0
            continue
        else:
            if '\t' not in line:
                tok = line[:-2]
                tag = line[-1]
            else:
                tok, tag = line.split('\t')
            assert (tag == 'B') | (tag =="I") | (tag =="O")
        
        if tag == 'B' and bosidx < 0:
            mention += tok + ' '
            bosidx = cur_bosidx
        elif tag == 'B' and bosidx >= 0:
            golden_spans.append((sample_idx, mention.strip(' '), bosidx, cur_bosidx-bosidx))
            mention, bosidx = tok + ' ', cur_bosidx
        elif tag == 'I' and bosidx < 0:
            pass
        elif tag == 'I' and bosidx >= 0:
            mention += tok + ' '
        elif tag == 'O' and bosidx < 0:
            pass
        elif tag == 'O' and bosidx >= 0:
            golden_spans.append((sample_idx, mention.strip(' '), bosidx, cur_bosidx-bosidx))
            mention, bosidx = '', -1
        else:
            RuntimeError('error in regexes')
        
        cur_bosidx += 1
    return golden_spans

def prepare_input_data(args):

    ## make two file 1. input data contains (sampleidx, mention, bosidx, span length) 2. label data contains (sampleidx, mention, bosidx, spanlength, code)
    
    if args.test_annot == 'all':
        annot = 'all_kb_kebiolm'
    elif args.test_annot == 'diffset':
        annot = f'diffset_kb_kebiolm_{args.subkb}'
    elif args.test_annot == 'subset':
        annot = f'subset_kb_kebiolm_{args.subkb}'

    with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset/{annot}/{args.test_set}.tsv', 'r') as f:
        golden_lines = [line.strip('\n') for line in f.readlines()]
    
    golden_spans = parse_ner_result(golden_lines)
    
    with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset/{annot}/{args.test_set}.code', 'r') as f:
        code_lines = [json.loads(line.strip('\n')) for line in f.readlines()]

    codes = sum(code_lines, [])
    assert len(codes) == len(golden_spans)

    for i, item in enumerate(golden_spans):
        golden_spans[i] = item + (codes[i],)

    return golden_spans
    # with open(f'/platform_tech/yuanzheng/GENE2E/MM_dataset/{annot}/{args.test_set}_gold_span.pkl', 'wb') as f:
    #     pickle.dump(golden_spans, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_kb", type=str, required=True)
    parser.add_argument("--test_kb", type=str, required=True)
    parser.add_argument("--test_annot", type=str, required=True)
    parser.add_argument("--test_set", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--subkb", type=str, required=True)
    args = parser.parse_args()
    print('===================start eval============')
    gold_spans = prepare_input_data(args)
    main_postprocess(args, gold_spans)
    # main_overall(args, gold_spans)
    # main_correct_ner(args)