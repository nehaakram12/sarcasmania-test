import flask
from flask import request, jsonify
import sys
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from random import randint


app = flask.Flask(__name__)
app.config["DEBUG"] = True

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

def init():
    # global d,loaded_model_humor,service
    # load the pre-trained Keras model
    print("Please wait while the Data-Models load!...")
    # d = []
    # dataFile = open('output1.txt', 'rb')
    # d = pickle.load(dataFile)
    # filename_humor = 'partial_fit_model.sav'
    # loaded_model_humor = pickle.load(open(filename_humor, 'rb'))



@app.route('/', methods=['GET'])
def home():
    return '''<h1>S.A.R.C.A.S.M.A.N.I.A</h1>
<p>Can you BEEEEEEEE more sarcastic??!!.</p><p>A prototype API for sarcasmania.</p>'''

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

@app.route('/api/sarcasmania', methods=['GET'])
def api_text():
    inputsen=""
    if 'text' in request.args:
        inputsen = (request.args['text'])
    else:
        return "Error: No text field provided. Please specify text."
    print("Input Line: ", inputsen)

    d = []
    dataFile = open('output1.txt', 'rb')
    print("File opened")
    d = pickle.load(dataFile)
    print("Dataset loaded")
    filename_humor = 'partial_fit_model.sav'
    loaded_model_humor = pickle.load(open(filename_humor, 'rb'))
    print("model loaded")
    t= create_tfidf_training_data_humor(d, inputsen)
    print("data created: ")
    lol = loaded_model_humor.predict(t)
    print("prediction calculated")
    humorscore = int(abs(int(lol[0])*100))

    results = {
     'Input': inputsen,
     'Humor': humorscore,
     'Insult': randint(0, 1)
    }

    return jsonify(results)


def create_tfidf_training_data_humor(docs, input):
    y = [d[0] for d in docs]
    corpus = [d[1] for d in docs]
    #corpus.append("helluss") #jo sentence hai wo yahan input dena ho ga for it to get vectorized
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    t=vectorizer.transform([input])
    return t


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print("* Loading model and Flask starting server...please wait until server has fully started")
    init()
    app.run(threaded=True)
