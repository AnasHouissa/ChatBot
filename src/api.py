# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:33:10 2023

@author: Houissa
"""


from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/getResponse', methods=['POST'])
def getResponse():
    # Get the input data from the request and call the get_response function
    response = {'response': get_response(request.get_json()['message'])}
    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    from prediction_model import get_response
    app.run()