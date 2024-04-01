import json
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
from openai import OpenAI


RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VERACITY_CLAIMS = 1
#VERACITY_MODEL = "LLM"
VERACITY_MODEL = "cross-encoder/nli-deberta-v3-base"


class Engine:
    def __init__(self):
        with open("data/corpus_evidence_unified.json", 'r') as j:
            self.evidence_database = list(json.load(j).values())
        self.tokenized_evidence = [e.split(" ") for e in self.evidence_database]
        self.bm25 = BM25Okapi(self.tokenized_evidence)
        self.rerank_xe = CrossEncoder(RERANKING_MODEL)
        if VERACITY_MODEL != "LLM":
            self.nli_xe = CrossEncoder(VERACITY_MODEL)
        else:
            self.client = OpenAI()

    def _top_100(self, claim):
        tokenized_claim = claim.split(" ")
        top = self.bm25.get_top_n(tokenized_claim, self.evidence_database, n=100)
        return top

    def _top_evidence(self, claim):
        evidence = self._top_100(claim)

        ranks = self.rerank_xe.rank(claim, evidence)
        top = [evidence[i] for i in [r["corpus_id"] for r in list(ranks)[:VERACITY_CLAIMS]]]
        return top

    def verify(self, claim):
        # TODO: putem lua mai multe claimuri in considerare cumva
        evidence = self._top_evidence(claim)

        if VERACITY_MODEL != "LLM":
            scores = self.nli_xe.predict([[claim, ev] for ev in evidence])

            label_mapping = ["contradiction", "entailment", "neutral"]
            labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
        else:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user",
                     "content": 'SETUP: I will give you one snippet that may or may not be relevant to a numerical '
                                'claim. You will have to decide whether this claim is supported ("entailment"), '
                                'contradicted ("contradiction") or unrelated/ambiguous/conflicted ("neutral") by the '
                                'evidence. Take into the account only the evidence that is relevant to the claim. '
                                'Reply only with "entailment" or "contradiction" or "neutral", with the meaning '
                                'stated above.'
                                'RUN:'
                                f'claim: {CLAIM}'
                                f'evidence: {evidence[0]}'}
                ]
            )

            labels = [completion.choices[0].message.content]

        return labels


CLAIM = ("More than 50 percent of immigrants from (El Salvador, Guatemala and Honduras) use at least one major welfare "
         "program once they get here")

e = Engine()

verdict = e.verify(CLAIM)
print(verdict)
