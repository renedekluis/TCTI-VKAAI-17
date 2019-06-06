from neuron import *
from hidden_layer import *
import copy


class NeuralNetwork:
	def __init__(self, name, numInputs, hiddenLayers, numOutputs, learnRate):
		self.name         	 	= name
		self.numInputs    	 	= numInputs
		self.hiddenLayerPattern = hiddenLayers
		self.numOutputs   	 	= numOutputs
		self.learnRate 		 	= learnRate
		self.hiddenLayers 	 	= []
		self.outputErrors		= []
		
		
		#ammountInputs 	= numInputs
		ammountOutputs 	= numOutputs
		#print(hiddenLayers)
		for idx, ammountNeurons in enumerate(hiddenLayers):	
			if idx < len(hiddenLayers)-1:
				if idx == 0:
					self.hiddenLayers.append(HiddenLayer(('H'+str(idx)), self.numInputs, ammountNeurons, hiddenLayers[idx+1], learnRate))
				else:
					self.hiddenLayers.append(HiddenLayer(('H'+str(idx)), hiddenLayers[idx-1], ammountNeurons, hiddenLayers[idx+1], learnRate))
			else:
				# Connected to Output Layer
				self.hiddenLayers.append(HiddenLayer(('H'+str(idx)), hiddenLayers[idx-1], ammountNeurons, self.numOutputs, learnRate))
		
		
		inputs = hiddenLayers[-1]
		self.hiddenLayers.append(HiddenLayer('Out',inputs,numOutputs,0,0.1))

		#for layer in self.hiddenLayers:
		#	layer.Show()
			

	
	
	
	
	def FeedForward(self, firstInput):
		inputs = firstInput
		for layer in self.hiddenLayers:
			inputs = layer.FeedForward(inputs);
		return inputs
	
	
	def GetLastKnownInputs(self):
		result = []
		for layer in self.hiddenLayers:
			result.append(layer.GetLastKnownInputs());
		return result
	
	def GetWeights(self):
		result = []
		for layer in self.hiddenLayers:
			result.append(layer.GetWeights());
		return result
	
	def CalcError(self, layerWeights, lastErrorDelta):
		
		errorDeltas = []
		#print('CalcError - ',layerWeights,lastErrorDelta)
		for delta in lastErrorDelta:
			for idx, neuronWeights in enumerate(layerWeights):
				tot = 0
				for weight in neuronWeights:
					tot += weight*delta
				errorDeltas.append(tot)
		#print('CalcError - ',errorDeltas)
		return errorDeltas
	
	
	def Train(self, inputList, OutputList):
		if len(inputList) is not self.numInputs or len(OutputList) is not self.numOutputs:
			print('Incorrect ammout of inputs or outputs')
			return []
		
		self.FeedForward(inputList)
		allInputs = self.GetLastKnownInputs()
		allWeights = self.GetWeights()

		newWeights = []
		self.outputErrors = []
		lastDeltaError = []
		for layerIdx, hiddenLayer in reversed(list(enumerate(self.hiddenLayers))):
			if hiddenLayer.name == "Out":
				newWeights = [hiddenLayer.Train(allInputs[-1],OutputList)] + newWeights
				lastDeltaError = hiddenLayer.GetDeltaErrors()
			else:
				lastDeltaError = self.CalcError(allWeights[layerIdx],lastDeltaError)
				newWeights = [hiddenLayer.Train(allInputs[layerIdx],lastDeltaError)] + newWeights
		
		for hiddenLayer in self.hiddenLayers:
			hiddenLayer.UpdateWeights()
		
		return newWeights
	
	
	
	
	
	
	
	
	
	
	def GetWeightsToOutputLayer(self):
		weights = []
		for idx, output in enumerate(self.outputNeurons):
			weights.append(output.GetWeights())
		return weights

		
		
	def TrainOutputLayer(self, allInputs, OutputList):
		#print('Training Output - inputs:',allInputs,'\tOutputs',OutputList)
		self.outputErrors = []
		newWeights = []
		for idx, output in enumerate(OutputList):
			newWeights = self.outputNeurons[idx].Train(allInputs[-1], output)
			self.outputErrors.append(self.outputNeurons[idx].GetError())
		return newWeights

		
		
	def GetInputValues(self, inputList):
		result = []
		if len(self.hiddenLayers) > 0:
			for hiddenLayer in self.hiddenLayers:
				result.append(hiddenLayer.GetLastKnownInputs())
		for out in self.outputNeurons:
			result.append(out.GetLastKnownInputs())
		return result
	
	
	def Train2(self, inputList, OutputList):
		if len(inputList) is not self.numInputs or len(OutputList) is not self.numOutputs:
			print('Incorrect ammout of inputs or outputs')
			return []
		
		self.Run(inputList)
		
		allInputs = self.GetInputValues(inputList)

		newOutputWeights = self.TrainOutputLayer(allInputs, OutputList)
		
		#print('\n\n\nSTART TRAINING HIDDEN LAYERS\n')
		weightsToNextLayer = self.GetWeightsToOutputLayer()
		lastError = self.outputErrors
		for layerIdx, hiddenLayer in reversed(list(enumerate(self.hiddenLayers))):
			#print('CLASS NeuralNetwork',lastError)
			if layerIdx == len(self.hiddenLayers)-1:
				weightsToNextLayer = hiddenLayer.Train(allInputs[layerIdx], weightsToNextLayer, lastError, self.outputNeurons, True)
				lastError = hiddenLayer.GetDeltaError()
			else:
				weightsToNextLayer = hiddenLayer.Train(allInputs[layerIdx], weightsToNextLayer, lastError, self.hiddenLayers[layerIdx+1])
				lastError = self.hiddenLayers[layerIdx+1].GetDeltaError()
			
		for hiddenLayer in self.hiddenLayers:
			hiddenLayer.UpdateWeights()

	
	def Run(self, inputList):
		input = inputList
		self.lastKnownInputs = [inputList]
		if len(self.hiddenLayers) > 0:
			for hiddenLayer in self.hiddenLayers:
				input = hiddenLayer.Run(input)
				self.lastKnownInputs.append(input)
		
		result = []
		for output in self.outputNeurons:
			result.append(output.Run(input))
		self.lastKnownInputs.append(result)
		return result
		

	def Show(self):
		for layer in self.hiddenLayers:
			layer.Show()
			




		