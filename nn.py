import numpy as np
import random
from util import oneHotEncodeY
import itertools

class NeuralNetwork:

    def __init__(self, out_nodes, lr):
        # Method to initialize a Neural Network Object
        # Parameters
        # out_nodes - number of output nodes
        # alpha - learning rate
        # batchSize - Mini batch size
        # epochs - Number of epochs for training
        self.layers = []
        self.out_nodes = out_nodes
        self.alpha = lr

    def addLayer(self, layer):
        # Method to add layers to the Neural Network
        self.layers.append(layer)


    def computeLoss(self, Y, predictions):
        # Returns the crossentropy loss function given the prediction and the true labels Y
        # TODO 
        # divide by minibatchsize
        # just the function, not the derivative
        # what are predictions ?
        # raise NotImplementedError

        n=Y.shape[0]
        # print(np.shape(predictions[-1]))
        # Loss=np.sum(np.sum(-1*(np.dot(Y,np.log(predictions[-1])) + np.dot(1-Y,np.log(1-predictions[-1]))))) 
        # Loss=np.sum(-1*(np.dot(Y,np.log(predictions[-1]))))
        val = np.clip(predictions[-1],1e-10,1-1e-10)
        Loss=np.sum(-1*(Y * np.log(val) + (1-Y) * np.log(1-val)))
        # Loss=np.sum(-1*(Y * np.log(predictions[-1]))) # matmul and dot are same and different from * and multiply  
        Loss=Loss/n
        return Loss

        # epsilon = 1e-10
        # return -1 * np.sum(np.sum(Y * np.log(np.clip(predictions, epsilon, 1. - epsilon))))


        

        # END TODO
    def computeAccuracy(self, Y, predLabels):
        # Returns the accuracy given the true labels Y and predicted labels predLabels
        correct = 0
        for i in range(len(Y)):
            if np.array_equal(Y[i], predLabels[i]):
                correct += 1
        accuracy = (float(correct) / len(Y)) * 100
        return accuracy

    def validate(self, validX, validY):
        # Input 
        # validX : Validation Input Data
        # validY : Validation Labels
        # Returns the validation accuracy evaluated over the current neural network model
        valActivations = self.feedforward(validX)
        pred = np.argmax(valActivations[-1], axis=1)
        validPred = oneHotEncodeY(pred, self.out_nodes)
        validAcc = self.computeAccuracy(validY, validPred)
        return pred, validAcc

    def feedforward(self, X):
        # Input
        # X : Current Batch of Input Data as an nparray
        # Output
        # Returns the activations at each layer(starting from the first layer(input layer)) to 
        # the output layer of the network as a list of np multi-dimensional arrays
        # Note: Activations at the first layer(input layer) is X itself     
        # TODO
        Activation_list=[X]
        for layer in self.layers:
            l=len(Activation_list)
            input_val=Activation_list[l-1]
            output_val=layer.forwardpass(input_val)
            Activation_list.append(output_val)
            # print(layer, " feedforward fine")

        return Activation_list
        # raise NotImplementedError
        # END TODO

    def backpropagate(self, activations, Y):
        # Input
        # activations : The activations at each layer(starting from second layer(first hidden layer)) of the
        # neural network calulated in the feedforward pass
        # Y : True labels of the training data
        # This method adjusts the weights(self.layers's weights) and biases(self.layers's biases) as calculated from the
        # backpropagation algorithm
        # Hint: Start with derivative of cross entropy from the last layer

        # TODO
        n=Y.shape[0]
        # print(Y)
        last_actv=activations[-1]
        val = np.clip(last_actv,1e-10,1-1e-10)
        delta = -1*(np.divide(Y,val) - np.divide(1-Y,1-val))
        # delta = -1*(np.divide(Y,last_actv))

        # delta=delta/n
        
        l=len(self.layers)
        # print(l-1)
        for i in range(l-1,-1,-1):
            # print(i)
            layer=self.layers[i]
            actv_prev=activations[i]
            delta = layer.backwardpass(self.alpha,actv_prev,delta) # LHS delta: delta[i-1], RHS delta: delta[i]
            # print(layer, " backwardpass fine")

        # i = len(activations) - 1
        # for layer in reversed(self.layers):
        #     # layer=layers[i]
        #     print(i)
        #     actv_prev=activations[i-1]
        #     delta = layer.backwardpass(self.alpha,actv_prev,delta) # LHS delta: delta[i-1], RHS delta: delta[i]
        #     i=i-1
        # raise NotImplementedError
        # END TODO