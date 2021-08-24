"""
Script for sentiment analysis model. For sentiment analysis, a Roberta transformers model pretrained on tweets sentiment
is used.
More info about transformers model: https://arxiv.org/abs/1706.03762
More info about the pretrained model: https://hf.co/cardiffnlp/twitter-roberta-base-sentiment.
"""

# importing libraries and modules
import preprocessor as p
from typing import Dict, Union
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# set options for the text preprocessor (More info: https://github.com/s/preprocessor)
p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)


# Sentiment Analyzer Class
class SentimentAnalyzer:
    def __init__(self):
        # Label mappings for the sentiment labels
        self.__sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}

        # Load transformers tokenizer and model. The text input is first needed to be tokenized using tokenizer before
        # passing the transformers model.
        self.__tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.__model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def __preprocess_text(self, text) -> Dict:
        """
        Given a text, this function will return the tokenized text after passing it through the tokenizer. Tokenization
        is an important step before passing a text to the transformers model. It converts the text to a set of numbers
        which the transformers model understands.
        :param text: The text that needs to be preprocessed and tokenized
        :return: Tokenized text
        """
        preprocessed_text = p.tokenize(text)
        encoded_input = self.__tokenizer(preprocessed_text, return_tensors='pt')
        return encoded_input

    def analyze(self, text: str) -> Union[Dict[str, float], None]:
        """
        Given a text, this function will perform sentiment analysis over it. Initially the text will be tokenized before
        passing it to the transformers model which gives a list of three scores. Softmax function is applied over the
        scores the normalize them to generate probability scores in the range of [0,1]
        :param text: The text for which the sentiment needs to be predicted.
        :return: A dictionary with the probability of each sentiment i.e. negative, neutral, positive
        """
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


# Sample main function to run the script independently
if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    sentiment = analyzer.analyze(text="""
    People simply want Cardano. I guarantee even ETH maximalists are buying $ADA. All are welcome, and all aboard!""")
    print(sentiment)
