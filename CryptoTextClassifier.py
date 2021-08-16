import pickle
import xgboost as xgb
import preprocessor as p
from typing import Dict, Union
from nltk.stem.porter import PorterStemmer

p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)


class CryptoTextClassifier:
    def __init__(self):
        self.__classification_labels = {0: 'Cherry Swap', 1: 'Cryptocurrency', 2: 'Defi'}
        with open('models/xgb_crypto.pkl', 'rb') as fin:
            self.__model = pickle.load(fin)
        with open('vectorizers/tfidf_crypto.pkl', 'rb') as fin:
            self.__vectorizer = pickle.load(fin)
        self.__stemmer = PorterStemmer()

    def __preprocess_text(self, text) -> str:
        preprocessed_text = p.tokenize(text).lower().split()
        preprocessed_text = [self.__stemmer.stem(i) for i in preprocessed_text]
        preprocessed_text = ' '.join(preprocessed_text)
        return preprocessed_text

    def classify(self, text) -> Union[Dict[str, float], None]:
        try:
            preprocessed_text = self.__preprocess_text(text)
            vectors = self.__vectorizer.transform([preprocessed_text])
            probas = self.__model.predict_proba(vectors)
            res_dict = dict()
            for i, proba in enumerate(probas):
                res_dict[self.__classification_labels[i]] = round(proba[0][1] * 100, 2)
        except:
            res_dict = {}
        return res_dict


if __name__ == '__main__':
    classifer = CryptoTextClassifier()
    classification = classifer.classify(
        text="""New airdrop live Cherry swap is a best project there community is very active We always with you  
        @Aman12300021 @AmanTriapthi @Mohit19337408 @CherryswapNet @OKExChain #cherryswap https://t.co/WFuc8gODCq""")
    print(classification)
