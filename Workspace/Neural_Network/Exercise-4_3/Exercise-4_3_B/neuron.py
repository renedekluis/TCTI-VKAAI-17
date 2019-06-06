import math
import random as r

class Neuron:
	def __init__(self, name, numInputs, learnRate):
		self.name = name
		self.weights   = []
		self.bias      = r.random()
		self.learnRate  = learnRate
		
		for i in range(numInputs):
			self.weights.append(r.random())
			
		
	def Show(self):
		print('='*10)
		print('Neuron: ', self.name)
		print('\tWeights:  ', self.weights   )
		print('\tBias:     ', self.bias      )
		print('\tLearnRate:', self.learnRate )
		print('='*10)
		
	def Activate(self,inputs):
		if len(inputs) < len(self.weights):
			return 0
		total = 0
		for idx, input in enumerate(inputs):
			total += float(self.weights[idx])*float(input)
		return math.tanh(total)	

	
	def GetDericative(self,input):
		return 1-math.tanh(math.tanh(input))
	

	def Update(self, inputs, output):
		if len(self.weights) < len(inputs):
			return 
		
		inputs.append(1) #Add Bias to inputs
		self.weights.append(self.bias) #Add bias Weight
		
		currentOutputValue = self.Activate(inputs)
		desiredOutput      = output
		n                  = self.learnRate
		newWeights         = []
		
		for idx, input in enumerate(inputs):
			W = self.weights[idx]
			dericative = self.GetDericative(input)
			newW = W + (n * input * dericative * (desiredOutput - currentOutputValue))
			newWeights.append(newW)
		
		self.weights = newWeights[:-1]
		self.bias    = newWeights[-1]
		inputs.pop()
		
		
	def Run(self, inputs):
		result = 0
		if len(self.weights) == len(inputs):
			inputs.append(1) #Append Bias as input
			self.weights.append(self.bias) #Append weight of bias
			result = self.Activate(inputs)
			self.weights = self.weights[:-1] #Remove bias
			inputs.pop()
		return result







































