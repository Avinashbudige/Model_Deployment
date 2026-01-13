# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os,gzip
# os.chdir('/home/sunil/Desktop/Great Lakes')
# Load the model
# model = pickle.load(open('finalized_model.sav','rb'))
# read data from a file
# Source - https://stackoverflow.com/a
# Posted by mishaF
# Retrieved 2026-01-13, License - CC BY-SA 3.0

with gzip.open('finalized_model.sav', 'rb') as ifp:
    model = pickle.load(ifp)

[1, 2, 3]


app = Flask(__name__)
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    predict_request=[[data['sl'],data['sw'],data['pl'],data['pw']]]
    predict_request=np.array(predict_request)
    print(predict_request)
    prediction = model.predict(predict_request)
    print(prediction)
    # Take the first value of prediction
    output = prediction[0]
    print(output)
    return jsonify(int(output))
 
if __name__ == '__main__':
    app.run(port=8111, debug=True)