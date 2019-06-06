import math
import random as r

class Neuron:
	def __init__(self, name, numInputs, numOutputs, learnRate):
		self.name = name
		self.weights   			= []
		#self.bias      			= r.random()
		self.bias      		= 0.5
		self.learnRate  		= learnRate
		self.newWeights         = []
		self.deltaError 		= 0
		self.lastKnownInputs 	= []
		self.lastKnownOutputs	= []
		self.numOutputs 		= numOutputs
		self.numInputs 			= numInputs
		
		for i in range(numInputs):
			self.weights.append(0.5)
			#self.weights.append(r.random())
			self.lastKnownInputs.append(0)
			
			
			
		
	def Show(self):
		print('\tNeuron:             ', self.name)
		print('\t\tWeights:          ', self.weights   )
		print('\t\tBias:             ', self.bias      )
		print('\t\tLearnRate:        ', self.learnRate )
		print('\t\tlastKnownInputs:  ', self.lastKnownInputs )
		print('\t\tlastKnownOutputs: ', self.lastKnownOutputs )
	
	
	
	def Check(self, inputs):
		return len(inputs) is len(inputs)
	
	
	
	def GetInput(self, inputs):
		inList = inputs[:]
		neuronWeightList = self.weights[:]
		inList.append(1)
		neuronWeightList.append(self.bias)
		result = 0
		for idx, i in enumerate(inList):
			result += i*neuronWeightList[idx]
		return result
	
	
	def Activate(self,inputTotal):
		return math.tanh(inputTotal)	
	
	
	def FeedForward(self, inputs):
		result = 0
		if not self.Check(inputs):
			print('Ammount inputs not equal to neuron inputs')
			return [0]
		
		totalInput = self.GetInput(inputs)
		self.lastKnownInputs = inputs;
		result = self.Activate(totalInput)
		self.lastKnownOutputs = result
		return result
	
	
	def GetLastKnownInputs(self):
		return self.lastKnownInputs
	
	
	def GetWeights(self):
		result = self.weights[:]
		result.append(self.bias)
		return result
	
	
	def GetDeltaError(self):
		return self.deltaError;
	
	
	
	def GetDericative(self,input):
		return 1-math.tanh(math.tanh(input))
	
	
		
	def BackPropagation(self, weight, input, desiredOutput, inputs):
		actualOutput = self.lastKnownOutputs
		dericative = self.GetDericative(self.GetInput(inputs))
		self.deltaError = dericative * (desiredOutput - actualOutput)
		return ( weight + (self.learnRate * input * self.deltaError) )

		
		
	def BackPropagation2(self, weight, activation, deltaError ):
		return weight + self.learnRate * activation * deltaError
	
	
	def Train(self, inputs, deltaOutput, isOutput = False):
		if len(self.weights) < len(inputs):
			print("Error: Inputs Not equal to Weight Ammount")
			return []
		
		inputs.append(1) #Add Bias to inputs
		self.weights.append(self.bias) #Add bias Weight
		self.newWeights         = []
		#print('Neuron - Train - ',inputs)
		for idx, input in enumerate(inputs):
			w = self.weights[idx]
			newW = 0.5
			if isOutput:
				newW = self.BackPropagation(w, input, deltaOutput, inputs)
			else:
				newW = self.BackPropagation2(w, input, deltaOutput)
			
			self.newWeights.append(newW)
		
		self.weights = self.weights[:-1]
		inputs.pop()
		return self.newWeights


	def UpdateWeights(self):
		self.weights = self.newWeights[:-1]
		self.bias    = self.newWeights[-1]
		
		



































