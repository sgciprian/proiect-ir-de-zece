import json
from engine import Engine


LIMIT_INSTANCES = 5
PRINT_INSTANCES = True

DATASET_PATH = "data/test_claims_quantemp.json"
EVIDENCE_PATH = "data/corpus_evidence_unified.json"

RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#VERACITY_MODEL = "LLM"
VERACITY_MODEL = "cross-encoder/nli-deberta-v3-base"

VERDICT_MAP = {
    "True": "entailment",
    "False": "contradiction",
    "Conflicting": "neutral"
}


e = Engine(EVIDENCE_PATH, RERANKING_MODEL, VERACITY_MODEL)

with open(DATASET_PATH, 'r') as j:
    data = json.load(j)

tests_max = min(len(data), LIMIT_INSTANCES)

tests_count = 0
tests_matching = 0
binary_true_positive = 0
binary_true_negative = 0
binary_false_positive = 0
binary_false_negative = 0

for instance in data:
    if tests_count >= LIMIT_INSTANCES:
        break

    if not PRINT_INSTANCES:
        work_done = tests_count / tests_max
        print("\rProgress: [{0:50s}] {1}%".format('#' * int(work_done * 50), work_done * 100), end="", flush=True)

    computed = e.verify(instance["claim"])
    computed_verdict = computed[0]
    real_verdict = VERDICT_MAP[instance["label"]]
    evidence = computed[1]

    tests_count += 1

    if PRINT_INSTANCES:
        print(f"{tests_count} / {tests_max}")
        print(f"Claim: {instance['claim']}")
        print(f"Computed: {computed_verdict}")
        print(f"Real: {real_verdict}")
        print(f"Evidence: {evidence}")
        print()

    if computed_verdict == real_verdict:
        tests_matching += 1
    if computed_verdict != "neutral" and real_verdict != "neutral":
        if computed_verdict == real_verdict:
            if computed_verdict == "entailment":
                binary_true_positive += 1
            elif computed_verdict == "contradiction":
                binary_true_negative += 1
        elif computed_verdict != real_verdict:
            if computed_verdict == "entailment":
                binary_false_positive += 1
            elif computed_verdict == "contradiction":
                binary_false_negative += 1

print(f"{tests_matching} correctly classified in {tests_count} test instances.")
print(f"")
print(f"{binary_true_positive + binary_true_negative + binary_false_positive + binary_false_negative} entailment"
      f"/contradiction labels")
print(f"{binary_true_positive} true positives")
print(f"{binary_true_negative} true negatives")
print(f"{binary_false_positive} false positives")
print(f"{binary_false_negative} false negatives")
