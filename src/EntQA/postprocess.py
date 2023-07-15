import pickle as pkl
import numpy as np
from reader import prune_predicts
from data_reader import load_data, get_golds, get_results_doc
from utils import Logger, compute_strong_micro_results


def evaluate_after_prune(logger, pruned_preds, golds,
                         samples):
    predicts_doc = get_results_doc(pruned_preds, samples)
    precision, recall, f_1 = compute_strong_micro_results(predicts_doc, golds,
                                                          logger)
    return {'precision': precision, 'recall': recall, 'F1': f_1}


def transform_predicts(preds, entities, samples):
    #  ent_idx,start,end --> start, end, ent name
    ent_titles = [e['title'] for e in entities]
    assert len(preds) == len(samples)
    results = []
    for ps, s in zip(preds, samples):
        results_p = []
        for p in ps:
            ent_title = ent_titles[s['candidates'][p[0]]]
            r = p[1:]
            # start, end, entity name
            r.append(ent_title)
            results_p.append(r)
        results.append(results_p)
    return results


if __name__ == "__main__":
    data_dir = "/mnt/data/run/BC5CDR_res/reader_input_shift/"
    kb_dir = "/mnt/data/Generative-End2End-IE/dataset/BC5CDR/kb/"
    with open("/mnt/data/run/BC5CDR_res/reader_results_shift/test_raw", "rb") as f:
        test_raw_predicts = pkl.load(f)

    data = load_data(data_dir, kb_dir)
    train_golds_doc, val_golds_doc, test_golds_doc, p_train_golds, \
    p_val_golds, p_test_golds = get_golds(data[0], data[1], data[2])
    dataset_map = {"test": 2, "val": 1}

    for thresd in 10.0**np.arange(-3, -1,  0.05):
        thresd = 0.01
        logger=Logger('log', on=False)

        pruned_test_preds = prune_predicts(test_raw_predicts, thresd)
        test_predicts = transform_predicts(pruned_test_preds, data[-1], data[2])
        test_result = evaluate_after_prune(logger, test_predicts,
                                           test_golds_doc, data[2])
        print(test_result)


