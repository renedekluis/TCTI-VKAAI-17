import numpy as np
import random as r

'''
A) Structure
------------
Implement the NOR-Gate using NumPy’s matrices and vectors. Use NumPy to
generate a random initial vector and create a truth table of the network output.

'''


norGate = [
	[[0,0], 1],
	[[0,1], 0],
	[[1,0], 0],
	[[1,1], 0]
]

theta 	= np.array([[r.random(), r.random()]])
bias 	= np.array([[r.random()]])


print('Theta: ', theta)
print('Bias:  ', bias)






