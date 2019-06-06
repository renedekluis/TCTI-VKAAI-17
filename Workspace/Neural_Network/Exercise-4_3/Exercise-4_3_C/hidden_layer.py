from neuron import *
import copy






	
class HiddenLayer:
	def __init__(self, name, numInputs, numNeurons, numOutputs, learnRate):
		self.name       = name
		self.numInputs  = numInputs
		self.numNeurons = numNeurons
		self.numOutputs = numOutputs
		self.neurons = []
		self.learnRate = learnRate
		self.allWeightsInLayer = []
		self.errors = []
		
		for idx in range(numNeurons):
			newNeuron = Neuron( (self.name+'-'+str(idx)), numInputs, numOutputs, learnRate )
			self.allWeightsInLayer.append(newNeuron.GetWeights())
			self.neurons.append(newNeuron)
			
		#print('weights in layer',self.name,':',self.allWeightsInLayer)
		
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
	
	
	
	
	def Train(self, inputList, outputList):
		newWeights = []
		print('HiddenLayer - Train - ',inputList,outputList,len(self.neurons))
		for neuronIdx, neuron in enumerate(self.neurons):
			
			if self.name == "Out":
				
				newWeights.append(neuron.Train(inputList[neuronIdx], outputList[neuronIdx], True))
			else:
				newWeights.append(neuron.Train(inputList[neuronIdx], outputList[neuronIdx]))
		
		#print('HiddenLayer - Train - ',newWeights)
		return newWeights
	
	




	def UpdateWeights(self):
		for neuronIdx, neuron in enumerate(self.neurons):
			neuron.UpdateWeights()


	




