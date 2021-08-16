import preprocessor as p
from typing import Dict, Union
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)


class SentimentAnalyzer:
    def __init__(self):
        self.__sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.__tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.__model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def __preprocess_text(self, text) -> Dict:
        preprocessed_text = p.tokenize(text)
        encoded_input = self.__tokenizer(preprocessed_text, return_tensors='pt')
        return encoded_input

    def analyze(self, text: str) -> Union[Dict[str, float], None]:
        try:
            encoded_input = self.__preprocess_text(text)
            output = self.__model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            res_dict = dict()
            for i, score in enumerate(scores):
                res_dict[self.__sentiment_labels[i]] = round(score * 100, 2)
        except:
            res_dict = {}
        return res_dict


if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    sentiment = analyzer.analyze(text="""
    People simply want Cardano. I guarantee even ETH maximalists are buying $ADA. All are welcome, and all aboard!""")
    print(sentiment)
