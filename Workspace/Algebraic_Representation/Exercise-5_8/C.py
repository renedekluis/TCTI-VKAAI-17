import numpy 	as np
import random 	as r
from backprop2 import *



'''
C) Training
----------------
Using the code from github (http://github.com/aldewereld/nl.hu.ict.a2i.cnn), 
train the network to correctly emulate the NOR-Gate.
In order to use the backpropagation algorithm provided, 
you need to adapt your predict function from above to a step-wise forward-function 
(see the description and function profile in the backprop.py-file on the provided git repository).

'''


inputs = [
	np.array([0, 0]),
	np.array([0, 1]),
	np.array([1, 0]),
	np.array([1, 1])
]

output = [
	np.array([1]),
	np.array([0]),
	np.array([0]),
	np.array([0])
]

w = [np.random.rand(1, 3)]



for x in range(len(inputs)):
	print('Expected: ', output[x], '\t Actual: ', forward(inputs[x], w))




print('\nTraining...\t', end='')
for i in range(200):
	for j in range(len(inputs)):
		differences = backprop(inputs[j], output[j], w)
		for index in range(len(w)):
			w[index] = w[index] + differences[index]
print('Done\n\n')




for x in range(len(inputs)):
	print('Expected: ', output[x], '\t Actual: ', forward(inputs[x], w)-1)




















