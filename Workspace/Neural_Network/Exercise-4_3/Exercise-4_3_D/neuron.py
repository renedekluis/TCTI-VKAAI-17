import math
import random as r

class Neuron:
	def __init__(self, name, numInputs, numOutputs, learnRate):
		self.name = name
		self.weights   			= []
		self.bias      			= r.random()
		#self.bias      		= 0.5
		self.learnRate  		= learnRate
		self.newWeights         = []
		self.delta 				= 0
		self.lastKnownInputs 	= []
		self.lastKnownOutputs	= []
		self.numOutputs 		= numOutputs
		self.numInputs 			= numInputs
		self.currDerr			= 0
		
		for i in range(numInputs):
			#self.weights.append(0.5)
			self.weights.append(r.random())
			self.lastKnownInputs.append(0)
		#print(self.name,self.weights)
			
			
		
	def Show(self):
		print('\tNeuron:             ', self.name)
		print('\t\tWeights:          ', self.weights   )
		print('\t\tBias:             ', self.bias      )
		print('\t\tLearnRate:        ', self.learnRate )
		print('\t\tlastKnownInputs:  ', self.lastKnownInputs )
		print('\t\tlastKnownOutputs: ', self.lastKnownOutputs )
		print('\t\tCurrent Derrecitive: ', self.currDerr )
	
	
	
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

		totalInput 				= self.GetInput(inputs)
		self.lastKnownInputs 	= inputs;
		result 					= self.Activate(totalInput)
		self.lastKnownOutputs 	= result
		return result
	
	
	def GetLastKnownInputs(self):
		return self.lastKnownInputs
	
	
	def GetWeights(self):
		result = self.weights[:]
		result.append(self.bias)
		return result
	
	
	def GetDeltaError(self):
		return self.delta;
	
	
	
	def GetDericative(self,input):
		d = 1-math.tanh(math.tanh(input))
		self.currDerr = d
		return d
	
	
		
	def BackPropagation(self, weight, activated, desiredOutput, inputs):
		actualOutput 		= self.lastKnownOutputs
		totInputs 			= self.GetInput(inputs)
		dericative 			= self.GetDericative(totInputs)
		self.delta 			= dericative * (desiredOutput - actualOutput)
		result = ( weight + (self.learnRate * activated * self.delta) )
		return result

		
		
	def BackPropagation2(self, weight, activation, deltaError ):
		result = (weight + self.learnRate * activation * deltaError)
		self.delta = deltaError
		return result
	
	
	def Train(self, inputs, deltaOutput, isOutput = False):
		if len(self.weights) < len(inputs):
			print("Error: Inputs Not equal to Weight Ammount")
			return []
		
		self.newWeights         = []
		for idx, input in enumerate(inputs):
			w = self.weights[idx]
			newW = 0.5
			if isOutput:
				newW = self.BackPropagation(w, input, deltaOutput, inputs)
			else:
				newW = self.BackPropagation2(w, input, deltaOutput)
			
			self.newWeights.append(newW)
		return self.newWeights


	def UpdateWeights(self):
		if 'H0' not in self.name:	
			self.weights = self.newWeights
			self.bias    = self.newWeights[-1]
		
		



































