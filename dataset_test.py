# Checks how many claims appear in the evidence dataset

import json


DATASET_PATH = "data/test_claims_quantemp.json"
EVIDENCE_PATH = "data/corpus_evidence_unified.json"

with open(DATASET_PATH, 'r') as j:
    data = json.load(j)

with open(EVIDENCE_PATH, 'r') as j:
    evid = list(["".join(s.lower().split()) for s in json.load(j).values()])

total = 0
identical = 0
identical_not_true = 0
contains = 0
contains_not_true = 0
for instance in data:
    claim = "".join(instance["claim"].lower().split())
    
    work_done = total / len(data)
    print("\rProgress: [{0:50s}] {1}%".format('#' * int(work_done * 50), work_done * 100), end="", flush=True)

    if any([claim == e for e in evid]):
        identical += 1
        if instance["label"] != "True":
            identical_not_true += 1
    if any([claim in e for e in evid]):
        contains += 1
        if instance["label"] != "True":
            contains_not_true += 1
    total += 1
   
print()
print(f"{identical} identical, {identical_not_true} not true, {contains} contained inside, {contains_not_true} not true | out of {total}")

