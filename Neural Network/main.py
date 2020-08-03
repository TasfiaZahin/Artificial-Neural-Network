import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import math
import operator
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import time

def relu(x):
    if(x > 0):
        return x
    else:
        return 0.0

def relu_prime(x):
    if(x > 0):
        return 1.0
    else:
        return 0.0

def tanh_func(x):
    return np.tanh(x)

def tanh_func_prime(x):
    ans = 1.0 - (np.tanh(x)*np.tanh(x))
    return ans

def sigmoid(a,x):
    den = 1.0 + math.exp(-a*x)
    ans = 1.0/den
    return ans

def sigmoid_prime(a,x):
    ans = a*sigmoid(1,x)*(1.0-sigmoid(1,x))
    return ans

class Neuron:

    def __init__(self,N):
        self.weights = [] #no of class length except first layer neurons
        self.yout = [] #N length
        self.del_val = [] #N length
        self.v = [] #N length
        self.e = [] #N length

        self.predicted_v = 0.0;
        self.predicted_yout = 0.0;

        for i in range(N):
            self.yout.append(0.0)
            self.v.append(0.0)
            self.del_val.append(0.0)
            self.e.append(0.0)


class NeuralNetwork:

    def __init__(self, layer_list, dataset, labels):

        self.N = len(dataset)
        self.no_of_features = len(dataset[0])
        self.dataset = dataset
        self.labels = labels
        self.no_of_layer = len(layer_list)
        self.network = {}

        for i in range(self.no_of_layer):
            length = layer_list[i]
            neurons = []
            for j in range(length):
                neurons.append(Neuron(self.N))
            self.network[i] = neurons

        layer = self.network[0] #layer zero has more input edges eqaul to feature no
        for i in range(len(layer)):
            neuron = layer[i]
            neuron.weights = np.random.rand(self.no_of_features + 1)


        for i in range(1,len(self.network)):
            prev_layer = self.network[i-1]
            layer = self.network[i]
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.weights = np.random.rand(len(prev_layer) + 1)


        # for i in range(len(self.network)):
        #     layer = self.network[i]
        #     for j in range(len(layer)):
        #         neuron = layer[j]
        #         print(neuron.weights)
        #     print()
        #print(self.network)
        # print(dataset)
        # print(labels)

    def getProductArray(self,array1,weights):

        sum = 0.0
        if(len(array1) == len(weights)-1):

            for i in range (len(array1)):
                sum += array1[i]*weights[i]

            sum += weights[len(weights)-1]

        else:
            print("array length not same")
        #print("sum:",sum)
        return sum


    def train_network(self,K,mew):

        #for count in range(K):
        c = 0
        while (1):

            c += 1
            print("Iteration: ",c)

            # print("old weights:")
            # for l in range(len(self.network)):
            #     print("new layer")
            #     layer = self.network[l]
            #     for neuron in layer:
            #         print(neuron.weights)

            for i in range(self.N):

                sample = self.dataset[i]
                label = self.labels[i]

                #calc v and yout

                layer0 = self.network[0]
                for neuron in layer0:
                    #print(neuron.weights)
                    #print(sample)
                    neuron.v[i] = self.getProductArray(sample,neuron.weights)
                    #print("v values 0:",neuron.v[0])

                for neuron in layer0:
                    neuron.yout[i] = sigmoid(1,neuron.v[i])

                for j in range(1,len(self.network)):

                    prev_layer = self.network[j-1]
                    prev_youts = []
                    for neuron in prev_layer:
                        prev_youts.append(neuron.yout[i])

                    current_layer = self.network[j]
                    for neuron in current_layer:
                        neuron.v[i] = self.getProductArray(prev_youts, neuron.weights)

                    for neuron in current_layer:
                        neuron.yout[i] = sigmoid(1,neuron.v[i])

            #calc cost function
            total_error = 0.0
            for i in range(self.N):

                sample = self.dataset[i]
                label = self.labels[i]

                layer_last = self.network[len(self.network) - 1]

                error_square = 0.0
                for j in range(len(layer_last)):
                    neuron = layer_last[j]
                    error_square += (neuron.yout[i] - label[j])*(neuron.yout[i] - label[j])
                    err = error_square/2.0

                total_error += err
            print("Cost function value: ",total_error)

            if(total_error <= 0.2 or c == K):
                break


            #calc e for backward calc
            for i in range(self.N):

                sample = self.dataset[i]
                label = self.labels[i]

                layer_last = self.network[len(self.network)-1]

                # err = 0.0
                # for j in range(len(layer_last)):
                #     neuron = layer_last[j]
                #     err += neuron.yout[i] - label[j]

                for j in range(len(layer_last)):
                    neuron = layer_last[j]
                    #neuron.e[i] = err
                    neuron.e[i] = neuron.yout[i] - label[j]
                    neuron.del_val[i] = neuron.e[i]*sigmoid_prime(1,neuron.v[i])
                    #print("label: ", label[j], " yout: ", neuron.yout[i], " err: ", neuron.e[i])

                for l in range(len(self.network)-2,-1,-1):

                    next_layer = self.network[l+1]
                    current_layer = self.network[l]

                    for j in range(len(current_layer)):

                        neuron = current_layer[j]

                        err = 0.0
                        for k in range(len(next_layer)):
                            nrn = next_layer[k]
                            err += nrn.del_val[i]*nrn.weights[j]

                        neuron.e[i] = err
                        neuron.del_val[i] = neuron.e[i]*sigmoid_prime(1,neuron.v[i])

            #update weights

            for l in range(len(self.network)-1,0,-1):

                current_layer = self.network[l]
                prev_layer = self.network[l-1]

                for j in range(len(current_layer)):
                    neuron = current_layer[j]
                    delta_w = []

                    for k in range(len(prev_layer) + 1):
                        delta_w.append(0.0)

                    for i in range(self.N):

                        youts = []
                        for nrn in prev_layer:
                           youts.append(nrn.yout[i])
                        youts.append(1)
                        #print("youtssssss append frist:", youts)

                        delj = neuron.del_val[i]
                        youts = delj*np.array(youts)
                        delta_w = np.add(delta_w,youts)

                    delta_w = -mew * np.array(delta_w)
                    neuron.weights = np.add(neuron.weights,delta_w)


            current_layer = self.network[0]

            for j in range(len(current_layer)):
                neuron = current_layer[j]
                delta_w = []

                for k in range(self.no_of_features + 1):
                    delta_w.append(0.0)

                for i in range(self.N):

                    youts = self.dataset[i]
                    youts = np.append(youts,1)
                    #print("youtssssss append:",youts)

                    delj = neuron.del_val[i]
                    youts = delj * np.array(youts)
                    delta_w = np.add(delta_w, youts)

                delta_w = -mew * np.array(delta_w)
                neuron.weights = np.add(neuron.weights, delta_w)


            # print("\nupdated weights:")
            # for l in range(len(self.network)):
            #     print("new layer")
            #     layer = self.network[l]
            #     for neuron in layer:
            #         print(neuron.weights)
            #
            # for j in range(len(self.network)):
            #     print("\nnew layer")
            #     layer = self.network[j]
            #     for neuron in layer:
            #         # print("printing vs")
            #         # print(neuron.v)
            #         print("printing youts")
            #         print(neuron.yout)
            #         # print("printing err")
            #         # print(neuron.e)
            #         print("printing delval")
            #         print(neuron.del_val)
            #         # print("printing weights")
            #         # print(neuron.weights)

        return self.network


    def predict(self, row):

        #print("\npredicting...")

        features = [] #separate the label

        for i in range(len(row)):
            features.append(row[i])

        #print(len(features))
        #print(features)

        layer0 = self.network[0]
        for neuron in layer0:
            #print(neuron.weights)
            neuron.predicted_v = self.getProductArray(features, neuron.weights)
            #print("v val: ",neuron.predcited_v)

        for neuron in layer0:
            neuron.predicted_yout = sigmoid(1,neuron.predicted_v)
            #print("yout val: ", neuron.predcited_yout)

        for j in range(1, len(self.network)):

            prev_layer = self.network[j - 1]
            prev_youts = []
            for neuron in prev_layer:
                prev_youts.append(neuron.predicted_yout)

            current_layer = self.network[j]
            for neuron in current_layer:
                neuron.predicted_v = self.getProductArray(prev_youts, neuron.weights)
                #print("v val: ", neuron.predicted_v)

            for neuron in current_layer:
                neuron.predicted_yout = sigmoid(1,neuron.predicted_v)
                #print("yout val: ", neuron.predicted_yout)

        final_youts = []
        last_layer = self.network[len(self.network)-1]

        for neuron in last_layer:
            final_youts.append(neuron.predicted_yout)

        #print("final youts:",final_youts)
        index, value = max(enumerate(final_youts), key=operator.itemgetter(1))
        #print(index+1, value)

        return (index+1)


def process_dataset_train(data):

    #print('preprocessing dataset...')
    dataset = data.copy() #pandas frame

    # norm = [0,1,2,3]
    #
    # for i in norm:
    #     mms = MinMaxScaler()
    #     dataset[[i]] = mms.fit_transform(dataset[[i]])

    X = dataset.drop(labels=dataset.columns[-1], axis=1)
    Y = dataset.iloc[:,-1]

    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, len(dataset.columns)-1].values

    X = preprocessing.scale(X)

    # add bias 1 to last column of X
    # X = pd.DataFrame(X)
    # X[len(X.columns)] = 1
    # X = X.values

    # dataset = dataset.values
    # label_encode = [len(dataset[0])-1]

    # for i in label_encode:
    #     labelencoder = LabelEncoder()
    #     dataset[:, i] = labelencoder.fit_transform(dataset[:, i])

        #onehotencoder = OneHotEncoder(categorical_features=[i])
        #dataset = onehotencoder.fit_transform(dataset).toarray()


    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)

    onehotencoder = OneHotEncoder(categories='auto')
    Y = onehotencoder.fit_transform(Y.reshape(-1,1)).toarray()

    print(X,Y)
    #
    # dataset = np.concatenate((X, Y), axis=1)

    #print(dataset)

    return X,Y

def process_dataset_test(data):

    #print('preprocessing dataset...')
    dataset = data.copy() #pandas frame

    norm = [0, 1, 2, 3]

    for i in norm:
        mms = MinMaxScaler()
        dataset[[i]] = mms.fit_transform(dataset[[i]])

    dataset = dataset.values
    #print(dataset)
    return dataset

def process_dataset_split(data):

    dataset = data.copy()  # pandas frame

    # norm = [0, 1, 2, 3]
    #
    # for i in norm:
    #     mms = MinMaxScaler()
    #     dataset[[i]] = mms.fit_transform(dataset[[i]])

    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, len(dataset.columns) - 1].values

    X = preprocessing.scale(X)

    print(X,Y)
    return X,Y

def main():

    dataset = pd.read_csv("trainNN.txt", delimiter='\t', header=None)
    X,Y = process_dataset_train(dataset)

    dataset = pd.read_csv("testNN.txt", delimiter='\t', header=None)
    test_X, test_Y = process_dataset_split(dataset)

    start = time.time()

    no_of_class = len(Y[0])
    layer_list = [2,no_of_class] #no of layers and neurons in each layer
    neural = NeuralNetwork(layer_list,X,Y)
    network = neural.train_network(6000,0.0001)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)


    error_count = 0;
    total = len(test_X)

    for i in range(len(test_X)):
        test_row = test_X[i]
        verdict = neural.predict(test_row)
        actual = test_Y[i]

        if(verdict != actual):
            error_count += 1
            print("actual: ", actual)
            print("verdict: ", verdict)

    acc = (total-error_count)/total*100.0

    print("\ntotal samples: ",total)
    print("number of misclassified: ",error_count)
    print("accuracy: ",acc,"%")
    print("Elapsed training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

main()