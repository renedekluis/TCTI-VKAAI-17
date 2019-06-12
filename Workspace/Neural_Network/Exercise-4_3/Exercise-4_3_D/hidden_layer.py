from neuron import *
import copy






	
class HiddenLayer:
	def __init__(self, name, numInputs, numNeurons, numOutputs, learnRate):
		self.name       = name
		self.numInputs  = numInputs
		self.numNeurons = numNeurons
		self.numOutputs = numOutputs
		self.neurons 	= []
		self.learnRate 	= learnRate
		self.errors 	= []
		self.allWeightsInLayer = []

		
		for idx in range(numNeurons):
			newNeuron = Neuron( (self.name+str(idx)), numInputs, numOutputs, learnRate )
			self.allWeightsInLayer.append(newNeuron.GetWeights())
			self.neurons.append(newNeuron)
		
		
	def Show(self):
		print('Name:',self.name)
		print('\tnumInputs',self.numInputs)
		print('\tnumOutputs',self.numOutputs)
		for neuron in self.neurons:
			neuron.Show()
	
	
	
	
	def FeedForward(self, input):
		output = []
		for neuron in self.neurons:
			output.append(neuron.FeedForward(input))
		return output
	
	
	
	def GetLastKnownInputs(self):
		result = []
		for neuron in self.neurons:
			result.append(neuron.GetLastKnownInputs())
		return result
	
	
	
	def GetWeights(self):
		result = []
		for neuron in self.neurons:
			result.append(neuron.GetWeights())
		return result
	
	
	
	def GetDeltaErrors(self):
		result = []
		for neuron in self.neurons:
			result.append(neuron.GetDeltaError())
		return result
	
	
	def TrainHidden(self, layerWeights, lastErrorDelta):
		errorDeltas = []
		for delta in lastErrorDelta:
			allInputs = []
			newWeights = []
			for idx, n in enumerate(self.neurons):
				sumOfError = 0
				nInVals = n.lastKnownInputs[:]
				nWeights = n.weights[:]
				nWeights.append(n.bias)
				for idx, w in enumerate(nWeights):
					sumOfError += w*delta
				input 		= n.GetInput(nInVals)
				allInputs.append(nInVals)
				dericative 	= n.GetDericative(input)
				deltaError 	= dericative*sumOfError
				newWeights.append(n.Train(nInVals, deltaError))
				errorDeltas.append(deltaError)
		return errorDeltas
	
	
	def Train(self, inputList, outputList):
		newWeights = []
		for neuronIdx, neuron in enumerate(self.neurons):
			if self.name == "Out":
				newWeights.append(neuron.Train(inputList[neuronIdx], outputList[neuronIdx], True))
			else:
				newWeights.append(neuron.Train(inputList[neuronIdx], outputList[neuronIdx]))
		
		return newWeights
	
	




	def UpdateWeights(self):
		for neuronIdx, neuron in enumerate(self.neurons):
			neuron.UpdateWeights()


	




