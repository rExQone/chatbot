import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    info = json.load(file)

try:
    with open("info.pickle", "rb") as f:
        words, labels, x_train, y_train = pickle.load(f)
except:
    #variable lists

    words = []
    labels = []
    x_docs = []
    docs_y = []
    # symbols_and_num = ['{', '}', '(', ')','[',']', '.', ',', ':', ';', '+', '-', '*', '/', '&', '|', '<', '>', '=','~'$', '1', '2', '3','4', '5', '6', '7', '8', '9', '0']

    #tokenization
    for intent in info["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)

            words.extend(wrds)

            x_docs.append(wrds)
            
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']

    print(words)
    words = sorted(list(set(words)))

    labels = sorted(labels)

    x_train = []
    y_train = []

    empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(x_docs):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output = empty[:]
        output[labels.index(docs_y[x])] = 1

        x_train.append(bag)
        y_train.append(output)


    x_train = numpy.array(x_train)
    y_train = numpy.array(y_train)

    with open("info.pickle", "wb") as f:
        pickle.dump((words, labels, x_train, y_train), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(x_train[0])])
net = tflearn.fully_connected(net, 16, activation = "relu")
net = tflearn.fully_connected(net, 16, activation = "relu")
net = tflearn.fully_connected(net, len(y_train[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load()
except:
    model.fit(x_train, y_train, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Conversation")
    print("type 'quit' for end session")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Thank you... I wish we will catch up later");
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in info["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()