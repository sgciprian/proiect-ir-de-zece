import json
from engine import Engine
import torch


DATASET_PATH = "data/test_claims_quantemp.json"
EVIDENCE_PATH = "data/corpus_evidence_unified.json"

RERANKING_MODELS = ["none", "cross-encoder/ms-marco-MiniLM-L-6-v2", "mixedbread-ai/mxbai-rerank-large-v1"] 
VERACITY_MODEL = "cross-encoder/nli-deberta-v3-base" # "LLM"
VERACITY_TEST_TYPE = "Top1" # "Top1", "Top5"

VERDICT_MAP = {
    "True": "entailment",
    "False": "contradiction",
    "Conflicting": "neutral"
}


with open(DATASET_PATH, 'r') as j:
    data = json.load(j)

iii = 0
for instance in data:
    computed = []
    for rr in RERANKING_MODELS:
        torch.cuda.empty_cache()
        e = Engine(EVIDENCE_PATH, rr, VERACITY_MODEL, VERACITY_TEST_TYPE)
        computed.append(e.verify(instance["claim"]))
        torch.cuda.empty_cache()
    computed_verdicts = [c[0] for c in computed]
    real_verdict = VERDICT_MAP[instance["label"]]
    evidences = [c[1] for c in computed]
    correct = [real_verdict == cv for cv in computed_verdicts]
    
    print(iii)
    iii += 1
    print(f"Claim: {instance['claim']}")
    print(f"Real: {real_verdict}")
    for i in range(len(RERANKING_MODELS)):
        print(f"RR Model: {RERANKING_MODELS[i]}")
        print(f"Computed: {computed_verdicts[i]}")
        print(f"Evidence: {evidences[i]}")
        print()
    print(flush=True)

