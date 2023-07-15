import json

umls_kb = json.load(open("UMLS/st21pv_kb.json"))

subsets = {
        "umls_t038": "umls_subset_T038.json",
        "umls_t058": "umls_subset_T058.json",
        "umls_snomed": "umls_subset_snomed.json"
        }
for name, path in subsets.items():
    subset_key = set(json.load(open("dataset/"+path)).keys())
    umls_subset_kb = {key:value for key, value in umls_kb.items() if key in subset_key}
    json.dump(umls_subset_kb, open("UMLS/"+name+"_kb.json", "w"))
