"""
An API script written using Flask. Contains three endpoints:
    - /check_liveliness: checks whether the API is live and working
    - /analyze_sentiment: predicts the sentiment of a text
    - /classify_text: classifies a text to its corresponding class
"""
import json
from signal import signal, SIGPIPE, SIG_DFL
from flask import Flask, request, make_response

from TextClassifier import TextClassifier
from SentimentAnalyzer import SentimentAnalyzer

signal(SIGPIPE, SIG_DFL)  # don't throw exception on broken pipe
app = Flask(__name__)

sentiment_analyzer = SentimentAnalyzer()
defi_text_classifier = TextClassifier(model='defi')


# checks whether the API is live and working
@app.route("/check_liveliness", methods=['GET'])
def check_liveliness():
    dummy_text = "New airdrop live Cherry swap is a best project there community is very active."
    try:
        sentiment = sentiment_analyzer.analyze(dummy_text)
        classification = defi_text_classifier.classify(dummy_text)
        if sentiment and classification:
            return {"status": "ok"}, 200
        raise Exception("Internal Error")
    except:
        return {"status": "failure"}, 500


# get sentiment of text
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        if request.headers['Content-Type'] == 'application/json':
            text = request.json.get('text')
            sentiment = sentiment_analyzer.analyze(text)
            response = make_response(json.dumps(sentiment))
            response.headers['Content-Type'] = 'application/json'
            response.status_code = 200
            return response
    except Exception as e:
        print("Error occurred: ", e)
        return {}, 400


# classify a text
@app.route('/classify_text', methods=['POST'])
def classify_text():
    try:
        if request.headers['Content-Type'] == 'application/json':
            text = request.json.get('text')
            clf = defi_text_classifier.classify(text)
            response = make_response(json.dumps(clf))
            response.headers['Content-Type'] = 'application/json'
            response.status_code = 200
            return response
    except Exception as e:
        print("Error occured: ", e)
        return {}, 400


def run_api(host='127.0.0.1', port=5000, debug=False):
    try:
        app.run(host=host, threaded=True, port=port, debug=debug)
    except Exception:
        print("starting with default port....")
        try:
            app.run(host='0.0.0.0', threaded=True, port=port + 100, debug=debug)
        except Exception:
            print("error in starting server...")


if __name__ == '__main__':
    run_api(host='0.0.0.0', port=8000, debug=False)
