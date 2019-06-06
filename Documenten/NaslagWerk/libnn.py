
from helperfuncs import *
import copy
import itertools
import random
class Neuron():

    def __init__(self,weights = []):
        self.weights = weights
        self.length = len(self.weights)
        self.delta = None

    def __str__(self):
        wstr = ["weights : "+(str(x)+"\n") for x in self.weights]
        return concat(operator.add, wstr)

    def len(self):

        self.length = len(self.weights)

    def setweights(self, weights):

        self.weights = weights
        self.length = len(self.weights)

    def setActivation(self, s):
        self.activation = s

    def setDelta(self,s):
        self.delta = s

    def result(self, input):
        if len(input) is len(self.weights):
                return sum(zipWithMultiply(self.weights, input))
        raise ValueError('amount of weights and amount of input does not same amount')
        print(input, self.weights)


    def DeltaRule(self,learningInput, learningRate, desiredAnswer):
        base = G(self.result(learningInput))
        for i in range(self.length):
            self.weights[i] = self.weights[i] + learningRate * learningInput[i] * Gprime(base) *(desiredAnswer - base)
        self.len()
        return(self.weights)

class Network():

    def __init__(self,layer = [],next_layer = None):

        self.layer = layer
        self.next_layer = next_layer
        self.len = len(layer)

    def __str__ (self):
        layer_string = concat (operator.add,
                               [str(j)+ '\n' + str(i)  for i,j in
                                list(zip(self.layer,range(self.len)))])

        if not self.next_layer:
            return layer_string
        return layer_string + "\n" + str(self.next_layer)

    def setLearnRate(self,l):
        self.learn_rate = l

    def result(self,input):
        result = list(map(lambda x: x.result(input),self.layer))
        if self.next_layer is None:
            return result
        return self.next_layer.result(result)

    def setExtraLayer(self,next):
        if self.next_layer is None:
            self.next_layer = next
        else:
            self.next_layer.setExtraLayer(next)

    def threshold(self,input):
        for i in self.layer:
            temp = 0
            for x,y in zip(input,i.weights):
                temp += (x*y)
            i.setActivation(G(temp))
        new_input = []
        if self.next_layer is not None:
            for i in self.layer:
                new_input.append(i.activation)
            self.next_layer.threshold(new_input)
        return

    def delta(self,input = []):

        if self.next_layer is not None:
            self.next_layer.delta(input)
            for i in range(len(self.layer)):
                w = []
                d = []
                for j in self.next_layer.layer:
                    w.append(j.weights[i])
                    d.append(j.delta)
                temp = 0
                for x,y in zip(w,d):
                    temp+= (x*y)
                self.layer[i].setDelta((Gprime(self.layer[i].activation)) * temp)
            return
        if len(self.layer) != len(input):
            print(len(self.layer),len(input))
            raise ValueError('input failure',)
        for i in range(len(self.layer)):
            self.layer[i].setDelta((Gprime(self.layer[i].activation)) * (input[i]-self.layer[i].activation))

    def weight(self,learningrate,input = []):

        for i in self.layer:
            for j in range(len(i.weights)):

                i.weights[j] =round( i.weights[j] + (i.delta * learningrate * input[j]),2)
        if self.next_layer is not None:
            new_input = []
            for i in self.layer:
                new_input.append(i.activation)
            self.next_layer.weight(learningrate,new_input)
        else:
            return

    def learn(self,input=[],learn_rate=1,should_be=[]):

        self.threshold(input)
        self.delta(should_be)
        self.weight(learn_rate,input)

    def print_activations(self,layer=1):
        count = 0
        for i in self.layer:

            print(' layer = ',layer, 'neuron = ', count , ' threshold= ',i.activation)
            count +=1
        if self.next_layer:
           self.next_layer.print_activations(layer= layer+1)
        return

    def print_delta(self,layer=1):
        count = 0
        for i in self.layer:

            print(' layer = ',layer, 'neuron = ', count , ' delta = ',i.delta)
            count +=1
        if self.next_layer:
           self.next_layer.print_delta(layer= layer+1)
        return

    def print_weights(self,layer = 1):
        count = 0
        for i in self.layer:
            for j in i.weights:
                print(' layer = ',layer, 'neuron = ', count , ' gewicht = ',j)
            count +=1
        if self.next_layer:
           self.next_layer.print_weights(layer= layer+1)
        return


def createNeuron(amount, minWeight, maxWeight ):
    return( Neuron([random.uniform(minWeight, maxWeight) for _ in range( amount)]))

def createLayer(amount, weightAmount, minWeight, maxWeight):
    return Network([createNeuron(weightAmount, minWeight, maxWeight)for _ in range(amount)])

def createNetwork(layers, min, max, before = None):
    a = createLayer(layers[1],layers[0],min,max)
    for i in range(2,(len(layers))):
        a.setExtraLayer(createLayer(layers[i],layers[i-1],min,max))
    return a
