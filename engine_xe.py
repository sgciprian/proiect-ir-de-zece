import json
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")


class Engine:
    def __init__(self, evidence_path, reranking_model, veracity_model, veracity_test):
        with open(evidence_path, 'r') as j:
            self.evidence_database = list(json.load(j).values())
        self.tokenized_evidence = [e.split(" ") for e in self.evidence_database]
        self.bm25 = BM25Okapi(self.tokenized_evidence)

        self.rerank_xe = CrossEncoder(reranking_model)
        self.veracity_model = veracity_model
        self.nli_xe = CrossEncoder(veracity_model)


        self.veracity_test_size = int(veracity_test.split("Top")[1])

        self.label_mapping = ["contradiction", "entailment", "neutral"]

    def _remove_stopwords(self, claim):
        words = word_tokenize(claim)
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_text = " ".join(filtered_words)
        return filtered_text

    def _top_100(self, claim):
        tokenized_claim = claim.lower().split(" ")
        top = self.bm25.get_top_n(tokenized_claim, self.evidence_database, n=100)
        return top

    def _top_evidence(self, claim, size=1):
        filtered_claim = self._remove_stopwords(claim)
        evidence = self._top_100(filtered_claim)

        ranks = self.rerank_xe.rank(claim, evidence)
        top = ["\n".join([evidence[i] for i in [r["corpus_id"] for r in list(ranks)[:size]]])]
        return top

    def verify(self, claim):
        evidence = self._top_evidence(claim, size=self.veracity_test_size)


        scores = self.nli_xe.predict([[claim, ev] for ev in evidence])
        label = ([self.label_mapping[score_max] for score_max in scores.argmax(axis=1)][0], evidence[0])

        return label
