#!flask/bin/python
from flask import Flask, jsonify
import pickle
import nltk
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]

result = {
    'status': 'success'
}

vectorizer = CountVectorizer()
classifier = MultinomialNB()

@app.route('/hubble/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

@app.route('/hubble/api/v1.0/tasks/train', methods=['GET'])
def train():
    emails = pd.read_csv("emails.csv")
    # Check for duplicates and dop those rows
    emails.drop_duplicates(inplace=True)

    # Tokenization
    emails['tokens'] = emails['text'].map(lambda text:  nltk.tokenize.word_tokenize(text))  

    # Removing stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    emails['filtered_text'] = emails['tokens'].map(lambda tokens: [w for w in tokens if not w in stop_words])

    # Removing 'Subject:'
    emails['filtered_text'] = emails['filtered_text'].map(lambda text: text[2:])

    # Mails still have many special charater tokens which may not be relevant for spam filter, lets remove these
    # Joining all tokens together in a string
    emails['filtered_text'] = emails['filtered_text'].map(lambda text: ' '.join(text))

    # Removing apecial characters from each mail 
    emails['filtered_text'] = emails['filtered_text'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))

    #Lemmatization
    wnl = nltk.WordNetLemmatizer()
    emails['filtered_text'] = emails['filtered_text'].map(lambda text: wnl.lemmatize(text))

    # Bag of Words
    counts = vectorizer.fit_transform(emails['filtered_text'].values)
    
    # Naive Bayes Classifier  
    targets = emails['spam'].values
    classifier.fit(counts, targets)
    serialize("model.spamorham")
    return jsonify({'result': result})

@app.route('/hubble/api/v1.0/tasks/return/<text>', methods=['GET'])
def mirror(text):
    return text

@app.route('/hubble/api/v1.0/tasks/predict/<emailText>', methods=['GET'])
def predict(emailText):
    emailText = [emailText]
    jaffa = deserialize("model.spamorham")
    if type(jaffa) == str:
        predictResult = {
            'status': 'failure',
            'error': 'model not found'
        }
        return jsonify({'result': predictResult})
        
    count_vectorizer = jaffa["vec"]
    model = jaffa["model"]
    emailTexts_counts = count_vectorizer.transform(emailText)
    prediction = model.predict(emailTexts_counts)
    if (prediction[0] == 0):
        predictResult = {
            'status': 'success',
            'result': 'This text is NOT SPAM'
        }
        print('This text is NOT SPAM ')
        return jsonify({'result': predictResult})
    if (prediction[0] == 1):
        predictResult = {
            'status': 'success',
            'result': 'This text is SPAM'
        }
        print('This text is SPAM')
        return jsonify({'result': predictResult})
    
def serialize(fname):
    with open(fname, 'wb') as f:
        pickle.dump(vectorizer, f)
        pickle.dump(classifier, f) 

def deserialize(fname):
    try:
        with open(fname, 'rb') as f:
            vec = pickle.load(f)
            model = pickle.load(f)
            return {'vec': vec, 'model': model} 
    except Exception:
        return "hello"

if __name__ == '__main__':
    app.run(debug=True)