"""
Script for the text classifier model. For text classification, a XGBoost Model (https://arxiv.org/abs/1603.02754) is
used. This script is modular and can easily be used to classify texts from different domains using XGBoost classifier
model trained on that domains. The paths for the saved model is fetched from classifier_configs.py script.
"""

# importing libraries and modules
import pickle
import xgboost as xgb
import preprocessor as p
from typing import Dict, Union
from classifier_configs import model_configs
from nltk.stem.porter import PorterStemmer

# set options for the text preprocessor (More info: https://github.com/s/preprocessor)
p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)


# Text Classifier Class
class TextClassifier:
    def __init__(self, model: str):
        # fetch model_configs for the user-input domain model
        configs = model_configs.get(model)
        if not configs:
            raise ValueError("Invalid model passed to TextClassifier. Please make sure a valid model name is passed.")

        # Label mappings for the classification labels
        self.__classification_labels = configs['label_mappings']

        # Load the trained XGBoost model
        with open(configs['model_path'], 'rb') as fin:
            self.__model = pickle.load(fin)

        # Load the saved TFIDF Vectorizer
        with open(configs['vectorizer_path'], 'rb') as fin:
            self.__vectorizer = pickle.load(fin)

        # A word stemmer to perform text cleaning. More info: https://www.geeksforgeeks.org/introduction-to-stemming
        self.__stemmer = PorterStemmer()

    def __preprocess_text(self, text) -> str:
        """
        Function for preprocessing and cleaning the given text. Two steps are mainly applied here for cleaning:
        1. Lower-casing the text and 2. Stemming all the words of the text to get the root words.
        After text cleaning, the cleaned text is passed to the loaded TFIDF vectorizer to get its corresponding vector
        which can directly be fed into the XGBoost model.
        :param text: The text that needs to be preprocessed and cleaned
        :return: Vectorized text
        """
        preprocessed_text = p.tokenize(text).lower().split()
        preprocessed_text = [self.__stemmer.stem(i) for i in preprocessed_text]
        preprocessed_text = ' '.join(preprocessed_text)
        preprocessed_text = self.__vectorizer.transform([preprocessed_text])
        return preprocessed_text

    def classify(self, text) -> Union[Dict[str, float], None]:
        """
        Function for classifying a given text to its corresponding predicted class. The text is first cleaned and
        vectorized and then given as input to the XGBoost model. The model gives out a list of probabilities of the text
        belonging to each class. Each of the probability score is rounded and is converted to percentages.
        :param text: The text that needs to be classified.
        :return: A dictionary containing the probability of the text belonging to each class.
        """
        try:
            preprocessed_text = self.__preprocess_text(text)
            probas = self.__model.predict_proba(preprocessed_text)[0]
            res_dict = dict()
            for i, proba in enumerate(probas):
                res_dict[self.__classification_labels[i]] = round(proba * 100, 2)
        except:
            res_dict = {}
        return res_dict


# Sample main function to run the script independently. The defi classification model is used here as an example.
if __name__ == '__main__':
    classifer = TextClassifier(model='defi')
    classification = classifer.classify(
        text="""New airdrop live Cherry swap is a best project there community is very active We always with you  
        @Aman12300021 @AmanTriapthi @Mohit19337408 @CherryswapNet @OKExChain #cherryswap https://t.co/WFuc8gODCq""")
    print(classification)
