import json
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
from openai import OpenAI
from math import exp
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')


prompt = '''
You are an Assistant responsible for helping detect whether the retrieved document is relevant to the query. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved document is relevant to the query.

Query: 25,000 Soldiers Returned Shaurya Chakra Medals to Support Farmers
Document: """... 25,000 awardees of shaurya chakra have returned the medals in support of the farmers' agitation. this is completely false, utterly baseless and malicious in ..."""
Relevant: Yes 

Query: Video shows Afghanistan woman stoned to death in November 2021.
Document: """people being subjected to such Horrifying punishments without Knowing their Crimes. And the World is a Usual Silent Spectator, Talibans stoning a woman some where in Afghanistan."""
Relevant: Yes 

Query: Russian scientists are planning to launch the COVID-19 vaccine by mid-August 2020.
Document: """the destruction of the world trade center was the result of a controlled demolition, this would mean that part of what happened on september 11, 2001"""
Relevant: No 

Query: Ron DeSantis wants to raise the retirement age to 70.
Document: """... ron desantis wants to raise the retirement age to 70, because he has walked back that position he took ten years earlier. 78. ron desantis lifted ..."""
Relvant: Yes

Query: Facebook Is Donating $1 Each Time Photo Is Shared Of This Puppy's Tumor
Document: """1 mar 2017  mcdonald's 2017 focus will be on four pillars: menu innovation, store renovations, digital ordering and delivery."""
Relevant: No

Query: {query}
Document: """{document}"""
Relevant:
'''

class Engine:
    def __init__(self, evidence_path, veracity_model):
        with open(evidence_path, 'r') as j:
            self.evidence_database = list(json.load(j).values())
        self.tokenized_evidence = [e.split(" ") for e in self.evidence_database]
        self.bm25 = BM25Okapi(self.tokenized_evidence)

        self.rerank_xe = OpenAI(api_key="") # add key here
        self.veracity_model = veracity_model
        self.nli_xe = CrossEncoder(veracity_model)

        self.label_mapping = ["contradiction", "entailment", "neutral"]

    def _top_100(self, claim):
        tokenized_claim = claim.split(" ")
        top = self.bm25.get_top_n(tokenized_claim, self.evidence_database, n=100)
        return top
    
    # @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def document_relevance(self, query, document):
        response = self.rerank_xe.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "user", "content": prompt.format(query=query, document=document)}
            ],
            temperature=0.4,
            logprobs=True,
            logit_bias={7566: 1, 2360: 1},
            max_tokens=1,
        )
        return (
            query,
            document,
            response.choices[0].message.content,
            response.choices[0].logprobs.content[0].logprob,
        )
    
    def _top_evidence2(self, claim):
        filtered_claim = self.remove_stopwords(claim)
        evidence = self._top_100(filtered_claim)
        max_prob = -1000
        max_evidence = ""
        for document in evidence:
            try:
                query, doc, response, logprobs = self.document_relevance(claim, document)
                probability = exp(logprobs)
                if response == "No":
                    probability = probability * (-1) + 1
                if probability > max_prob: 
                    max_prob = probability 
                    max_evidence = doc
            except Exception as e:
                print(e)
        return [max_evidence]

    def remove_stopwords(self, claim):
        words = word_tokenize(claim)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_text = ' '.join(filtered_words)
        return filtered_text

    def verify(self, claim):
        evidence2 = self._top_evidence2(claim)
        scores = self.nli_xe.predict([[claim, ev] for ev in evidence2])
        label = ([self.label_mapping[score_max] for score_max in scores.argmax(axis=1)][0], evidence2[0])
        return label


