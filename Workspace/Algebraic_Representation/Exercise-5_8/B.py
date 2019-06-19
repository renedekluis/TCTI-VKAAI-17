import numpy 	as np
import random 	as r
import math


'''
B) Feed Forward Function
-----------------------------
Generalise your NOR-Gate to a function predict(x, Theta) that, given an input
vector x and a list of weight matrices [Q1;Q2; : : : ;Qn], predicts the associated y value.

'''


def Sigmoid(x):
    return 1 / (1 + math.e ** (-x))


def predict(x, theta):
	prediction = None
	for weight in theta:
		if prediction == None:
			prediction = Sigmoid(np.dot(weight, x))
		else:
			prediction = Sigmoid(np.dot(weight, prediction))
	return prediction



norGate = [
	[[0, 0, 0], 0],
	[[0, 0, 1], 1],
	[[0, 1, 0], 1],
	[[0, 1, 1], 0],
	[[1, 0, 0], 1],
	[[1, 0, 1], 0],
	[[1, 1, 0], 0],
	[[1, 1, 1], 1]
	]

theta = np.array([[r.random(), r.random(), r.random()]])


for option in norGate:
	print('Option: ',option[0],'Theta: ',theta,'Predicted:',predict(option[0], theta))























