import json
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
from openai import OpenAI


class Engine:
    def __init__(self, evidence_path, reranking_model, veracity_model):
        with open(evidence_path, 'r') as j:
            self.evidence_database = list(json.load(j).values())
        self.tokenized_evidence = [e.split(" ") for e in self.evidence_database]
        self.bm25 = BM25Okapi(self.tokenized_evidence)

        self.rerank_xe = CrossEncoder(reranking_model)
        self.veracity_model = veracity_model
        if veracity_model != "LLM":
            self.nli_xe = CrossEncoder(veracity_model)
        else:
            self.client = OpenAI()

        self.label_mapping = ["contradiction", "entailment", "neutral"]

    def _top_100(self, claim):
        tokenized_claim = claim.split(" ")
        top = self.bm25.get_top_n(tokenized_claim, self.evidence_database, n=100)
        return top

    def _top_evidence(self, claim):
        evidence = self._top_100(claim)

        ranks = self.rerank_xe.rank(claim, evidence)
        top = [evidence[i] for i in [r["corpus_id"] for r in list(ranks)[:1]]]
        return top

    def verify(self, claim):
        # TODO: putem lua mai multe claimuri in considerare cumva
        evidence = self._top_evidence(claim)

        if self.veracity_model != "LLM":
            scores = self.nli_xe.predict([[claim, ev] for ev in evidence])
            label = ([self.label_mapping[score_max] for score_max in scores.argmax(axis=1)][0], evidence[0])
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
                                f'claim: {claim}'
                                f'evidence: {evidence[0]}'}
                ]
            )

            label = (completion.choices[0].message.content, evidence[0])

        return label
