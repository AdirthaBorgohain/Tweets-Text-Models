"""
Script for training a new XGBoost Classifier model from excel data. Given an excel sheet as in the same format as
Defi_campaign.xlsx, the ClassifierTrainer class can preprocess and prepare the data from it to train a new XGBoost
Classifier Model (https://arxiv.org/abs/1603.02754). After training, the model and the fitted TFIDF vectorizer is saved
according to the user-given names.
"""

# importing libraries and modules
import pickle
import pandas as pd
import xgboost as xgb
import preprocessor as p
from typing import List, Tuple
from nltk.stem.porter import PorterStemmer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# set options for the text preprocessor (More info: https://github.com/s/preprocessor)
p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)


# Classifier Trainer Class
class ClassifierTrainer:
    def __init__(self, excel_file: str):
        # A word stemmer to perform text cleaning. More info: https://www.geeksforgeeks.org/introduction-to-stemming
        self.__stemmer = PorterStemmer()
        # A TFIDF vectorizer that assigns scores to each word present in the train data based on its importance and
        # frequency of occurrence in the training data. More info: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        self.__vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        # A MultiLabelBinarizer is used to generate the labels and tags for training the classifier model.
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
        self.__mlb = MultiLabelBinarizer()
        # Initializing the XGBoost Classifier Model
        self.__xgb_model = xgb.XGBClassifier(objective='binary:logistic')
        # The data from the excel file is cleaned and parsed as needed for training the XGBoost model.
        self.df = self.__process_file(excel_file)

    def __process_file(self, excel_file: str) -> pd.DataFrame:
        """
        processes the raw excel file to clean all the message texts as needed for training.
        :param excel_file:
        :return: Dataframe after processing is performed on it.
        """
        df = pd.read_excel(excel_file, engine='openpyxl')
        # Take only the columns necessary for training.
        df = df[['message', 'tags']]

        # Remove duplicates and remove rows with null values in the data, if any.
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)

        # Clean message texts by lower-casing the texts and stemming the words of the texts to keep only the root words.
        df['message_cleaned'] = df['message'].apply(lambda x: p.tokenize(x))
        df['message_cleaned'] = df['message_cleaned'].apply(lambda x: x.lower())
        tokenized_text = df['message_cleaned'].apply(lambda x: x.split())  # tokenizing
        tokenized_text = tokenized_text.apply(lambda x: [self.__stemmer.stem(i) for i in x])
        df['message_cleaned'] = tokenized_text.apply(lambda x: ' '.join(x))
        return df

    def __generate_sentences_labels(self, df: pd.DataFrame, target_cols: List[str]) -> Tuple[pd.DataFrame, List]:
        """
        This function extracts sentences and labels from the dataframe in the format which can be used for training the
        model.
        :param df: The processed Dataframe.
        :param target_cols: The categories or classes to which the model needs to learn to classify.
        :return: The extracted sentences along with its corresponding labels.
        """
        sentences = df['message_cleaned']
        df['tags'] = df['tags'].apply(lambda x: x.split(', '))

        # Separates out each of the labels and tags to a format that can be used to train the XGBoost Model
        df = df.join(pd.DataFrame(self.__mlb.fit_transform(df.pop('tags')),
                                  columns=self.__mlb.classes_,
                                  index=df.index))
        labels = df[target_cols].values
        return sentences, labels

    @staticmethod
    def __evaluate_model(model, X_test, y_test) -> None:
        """
        Checks accuracy of the XGBoost Classifier Model on test data and prints the results.
        :param model: The trained XGBoost model.
        :param X_test: Test data to make the predictions on
        :param y_test: Actual labels for the test data
        :return: None
        """
        print('Accuracy on test data: {:.1f}%'.format(accuracy_score(y_test, model.predict(X_test)) * 100))
        print('Classification Report: \n ', classification_report(y_test, model.predict(X_test)))

    def train_model(self, target_cols: List[str], model_name: str, vectorizer_name: str) -> None:
        """
         Trains a XGBoost Classifier model on the data from the excel file. Also, prints the accuracy of the trained model.
        :param target_cols: The categories or classes to which the model needs to learn to classify.
        :param model_name: Name of the file to which the trained model must be saved to.
        :param vectorizer_name: Name of the file to which the TFIDF vectorizer must be saved to.
        :return: None
        """
        sentences, labels = self.__generate_sentences_labels(self.df, target_cols)

        # Fit TFIDF vectorizer on the cleaned sentence texts
        self.__vectorizer.fit(sentences)

        # Save the fitted vectorizer
        with open(f'vectorizers/{vectorizer_name}', 'wb') as fout:
            pickle.dump(self.__vectorizer, fout)

        # Split the data to train set and test set
        sents_train, sents_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1, random_state=2021)

        # Using the fitted vectorizer, generate vectors for the train set and test set to train and test the model on.
        X_train, X_test = self.__vectorizer.transform(sents_train), self.__vectorizer.transform(sents_test)

        #  A OneVsRestClassifier classifier with XGBoost model is used to train since this is a multiclass problem and
        # hence the model is trained individually for each of the classes in the training data.
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
        xgb_estimator = self.__xgb_model
        multilabel_model = OneVsRestClassifier(xgb_estimator)
        multilabel_model.fit(X_train, y_train)

        # Save the trained model
        with open(f'models/{model_name}', 'wb') as fout:
            pickle.dump(multilabel_model, fout)
        print(f'Trained model successfully saved to "models/{model_name}"')

        # Perform evaluation on the trained model
        self.__evaluate_model(multilabel_model, X_test, y_test)


# Sample main function to run the script independently. A classification model is trained using the Defi_campaign.xlsx
# file here as an example.
if __name__ == '__main__':
    trainer = ClassifierTrainer(excel_file='data/Defi_campaign.xlsx')
    trainer.train_model(model_name='xgb_defi.pkl', vectorizer_name='tfidf_defi.pkl',
                        target_cols=['Cherry Swap', 'Cryptocurrency', 'Defi'])
