from neuron import *

twoInputOptions = [
	[0,0],
	[1,0],
	[0,1],
	[1,1] ]

	
weights   = [-0.5, -0.5]
threshold = 0
NOR       = Neuron(weights, threshold)

for option in twoInputOptions:
	print(option, NOR.Run(option))