"""
Contains the script for training a new classifier model from excel data.
"""
import pickle
import pandas as pd
import xgboost as xgb
from typing import List
import preprocessor as p
from nltk.stem.porter import PorterStemmer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)


class ClassifierTrainer:
    def __init__(self, excel_file: str):
        self.__stemmer = PorterStemmer()
        self.__vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.__mlb = MultiLabelBinarizer()
        self.__xgb_model = xgb.XGBClassifier(objective='binary:logistic')
        self.df = self.__process_file(excel_file)

    def __process_file(self, excel_file: str):
        df = pd.read_csv(excel_file, engine='openpyxl')
        df = df[['message', 'tags']]
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df['message_cleaned'] = df['message'].apply(lambda x: p.tokenize(x))
        df['message_cleaned'] = df['message_cleaned'].apply(lambda x: x.lower())
        tokenized_text = df['message_cleaned'].apply(lambda x: x.split())  # tokenizing
        tokenized_text = tokenized_text.apply(lambda x: [self.__stemmer.stem(i) for i in x])
        df['message_cleaned'] = tokenized_text.apply(lambda x: ' '.join(x))
        return df

    def __generate_sentences_labels(self, df: pd.DataFrame, target_cols: List[str]):
        df['tags'] = df['tags'].apply(lambda x: x.split(', '))
        df = df.join(pd.DataFrame(self.__mlb.fit_transform(df.pop('tags')),
                                  columns=self.__mlb.classes_,
                                  index=df.index))
        labels = df[target_cols].values
        return labels

    def __evaluate_model(self, model, X_test, y_test):
        print('Accuracy on test data: {:.1f}%'.format(accuracy_score(y_test, model.predict(X_test)) * 100))
        print('Classification Report: \n ', classification_report(y_test, model.predict(X_test)))

    def train_model(self, target_cols: List[str], model_name: str, vectorizer_name: str):
        sentences, labels = self.__generate_sentences_labels(self.df, target_cols)
        self.__vectorizer.fit(sentences)
        with open(f'vectorizers/{vectorizer_name}', 'wb') as fout:
            pickle.dump(self.__vectorizer, fout)
        sents_train, sents_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1, random_state=2021)
        X_train, X_test = self.__vectorizer.transform(sents_train), self.__vectorizer.transform(sents_test)
        xgb_estimator = self.__xgb_model
        multilabel_model = OneVsRestClassifier(xgb_estimator)
        multilabel_model.fit(X_train, y_train)
        with open(f'models/{model_name}', 'wb') as fout:
            pickle.dump(multilabel_model, fout)
        print(f'Trained model successfully saved to "models/{model_name}"')
        self.__evaluate_model(multilabel_model, X_test, y_test)


if __name__ == '__main__':
    trainer = ClassifierTrainer(excel_file='data/Defi_campaign.xlsx')
    trainer.train_model(model_name='xgb_crypto.pkl', vectorizer_name='tfidf_crypto.pkl')
