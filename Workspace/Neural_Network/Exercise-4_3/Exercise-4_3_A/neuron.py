

class Neuron:
	def __init__(self, weights, threshold):
		self.weights = []
		self.threshold = threshold
		for weight in weights:
			self.weights.append(weight)
		
		
	def Run(self, inputs):
		result = 0
		for idx, input in enumerate(inputs):
			result += inputs[idx] * self.weights[idx]

		result = 1 if result >= self.threshold else 0
		return result