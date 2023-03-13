# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:08:25 2023

@author: Houissa
"""


import json
# encoding and decoding JSON
import string
# collection of string constants and functions
import random
# generating random numbers, selecting random items from a sequence
import nltk
#nltk.download('omw-1.4')
# natural language processing (NLP) tasks,
# such as tokenization,
# stemming,
# lemmatization, and part-of-speech tagging.
import numpy as np
# multi-dimensional arrays
from nltk.stem import WordNetLemmatizer
# module for lemmatizing words using WordNet, a lexical database for the English language.
import tensorflow as tf
# building and training machine learning models, especially deep neural networks
from keras.models import Sequential
# building neural networks in a sequential manner, where each layer is added in sequence.
from tensorflow.keras.layers import Dense, Dropout
# dense specifying a fully connected layer in a neural network
# Dropout preventing overfitting in a neural network by randomly dropping out some neurons during training.

# nltk.download("punkt")
# library that contains pre-trained models for sentence tokenization in various languages.
# nltk.download("wordnet")
# WordNet lexical database, which is a semantic network of English words and their relationships.

with open('../dataset/dataset.json') as file:
    data = json.load(file)


patterns = []
tags = []
data_x = []
data_y = []
training = []


for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize the pattern into a list of words. divide it into tokesn ( words / characters)
        tokens = nltk.word_tokenize(pattern)
        # Add the individual words in the pattern to the "patterns" list.
        patterns.extend(tokens)
        # Add the entire pattern to the "data_x" list as input data.
        data_x.append(pattern)
        # Add the intent tag to the "data_y" list as output data.
        data_y.append(intent["tag"])
        # example
        #patterns = ['!', ',', '.', 'afternoon', 'doing', 'evening', 'friend', 'go', 'going', 'good', 'greeting', 'hey', 'hi', 'how', 'howdy', 'is', 'it', 'morning', 'nice', 'see', 'there', 'up', 'welcome', 'what', 'you']
        #tags = ['greetings']
        #data_x = ['Hi', 'Hey', 'Hello', 'Greetings', 'Good morning', 'Good afternoon', 'Good evening', 'Howdy', 'Hey there']
        #data_y = ['greetings', 'greetings', 'greetings', 'greetings', 'greetings', 'greetings', 'greetings', 'greetings', 'greetings']

    # If the intent tag is not already in the "tags" list, add it.
    if intent["tag"] not in tags:
        tags.append(intent["tag"])
# Create a WordNetLemmatizer object for lemmatizing words.
lemmatizer = WordNetLemmatizer()
# Lemmatize each pattern in the "patterns" list and remove any punctuation. doing => do
patterns = [lemmatizer.lemmatize(
    pattern.lower()) for pattern in patterns if pattern not in string.punctuation]
# Remove any duplicate patterns and sort the list alphabetically.
patterns = sorted(set(patterns))
# Remove any duplicate tags and sort the list alphabetically.
tags = sorted(set(tags))
# patterns = [afternoon', 'doing', 'evening', 'friend', 'go', 'going', 'good', 'greeting', 'hey', 'hi', 'how', 'howdy', 'is', 'it', 'morning', 'nice', 'see', 'there', 'up', 'welcome', 'what', 'you']
#tags = ['greetings']


# Create a list of zeros with length equal to the number of tags.
out_empty = [0] * len(tags)

for idx, doc in enumerate(data_x):
    bow = []  # Create an empty list for the bag of words.
    # Lemmatize the document and convert to lowercase. doing => do
    text = lemmatizer.lemmatize(doc.lower())
    # Loop through each pattern in the "patterns" list.
    for pattern in patterns:
        # If the pattern is in the document, append a 1 to the bag of words, otherwise append a 0.
        bow.append(1) if pattern in text else bow.append(0)
    # Create a list of zeros with length equal to the number of tags.
    output_row = list(out_empty)
    # Set the output row at the index of the current document's tag to 1.
    output_row[tags.index(data_y[idx])] = 1
    # Append the current bag of words and output row to the "training" list.
    training.append([bow, output_row])

random.shuffle(training)  # Shuffle the training data randomly.

# Convert the "training" list to a NumPy array with data type "object".
training = np.array(training, dtype=object)
# Extract the bag of words and output rows into separate NumPy arrays.
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


model = Sequential()  # Create a sequential model.
# Add a densely connected layer with 128 units, taking the length of a training example as input.
# Use the ReLU activation function.
model.add(Dense(512, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))  # Add a dropout layer with a rate of 0.5.
# Add another densely connected layer with 64 units and ReLU activation.
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))  # Add another dropout layer with a rate of 0.5.
# Add a final densely connected layer with units equal to the number of tags, using softmax activation.
model.add(Dense(len(train_y[0]), activation="softmax"))
# Define the Adam optimizer with specified parameters.
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',  # Compile the model with categorical cross-entropy loss.
              optimizer=adam,
              metrics=["accuracy"])  # Use accuracy as the evaluation metric.
# print(model.summary())  # Print a summary of the model.
# Train the model on the training data for 150 epochs, with verbose output.
model.fit(x=train_x, y=train_y, epochs=300, verbose=0)
# an epoch refers to one complete iteration over the entire dataset during training.


def clean_text(text):
    # Tokenize the text into individual words.
    tokens = nltk.word_tokenize(text)
    # Lemmatize each word to reduce inflectional forms to their base form.
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens  # Return the cleaned tokens.


def bag_of_words(text, vocab):
    # Clean the text by tokenizing and lemmatizing it.
    tokens = clean_text(text)
    bow = [0] * len(vocab)  # Initialize a bag of words vector with zeros.
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:  # If the current word is in the vocabulary, set the corresponding index in the bag of words vector to 1.
                bow[idx] = 1
    return np.array(bow)  # Return the bag of words vector as a numpy array.


def pred_tag(text, vocab, labels):
    # convert input text to bag of words representation
    bow = bag_of_words(text, vocab)
    # predict tag using the model
    result = model.predict(np.array([bow]))[0]
    thresh = 0.50
    # filter predictions above a certain threshold
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    # sort predictions by probability in descending order
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    # get predicted tag(s)
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_response(input_txt):
    intents = pred_tag(input_txt, patterns, tags)
    # if no tag is predicted
    if len(intents) == 0:
        result = "Sorry! I don't understand"
    # if a tag is predicted
    else:
        tag = intents[0]
        list_of_intents = data["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    return result
