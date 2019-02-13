from neuron import *


class LogicGates:
	def __init__(self):
		pass


	'''
	Brief: 
		This function runs a logic Nor.
	
	Parameters:
		inputs: List of inputs for the gate
	
	Return:
		Output of the logic gate
	'''	
	def LogicNor(self, inputs):
		weights = []
		for input in inputs:
			weights.append( -(1/len(inputs)) )
		return Neuron(weights, 0 ).Run(inputs)

	'''
	Brief: 
		This function runs a logic Or.
	
	Parameters:
		inputs: List of inputs for the gate
	
	Return:
		Output of the logic gate
	'''	
	def LogicOr(self, inputs):
		weights = []
		for input in inputs:
			weights.append( 0.5 )
		return Neuron(weights, 0.5 ).Run(inputs)

	'''
	Brief: 
		This function runs a logic And.
	
	Parameters:
		inputs: List of inputs for the gate
	
	Return:
		Output of the logic gate
	'''	
	def LogicAnd(self, inputs):
		weights = []
		for input in inputs:
			weights.append( (1/len(inputs)) )
		andGate = Neuron(weights,  1  )
		return andGate.Run(inputs)

	'''
	Brief: 
		This function runs a logic Nand.
	
	Parameters:
		inputs: List of inputs for the gate
	
	Return:
		Output of the logic gate
	'''	
	def LogicNand(self, inputs):
		weights = []
		treshold = 0
		for idx, input in enumerate(inputs):
			weights.append( -(1/len(inputs)) )
			if idx < len(inputs)-1:
				treshold += -(1/len(inputs))
		return Neuron(weights, treshold ).Run(inputs)
	
	'''
	Brief: 
		This function runs a logic Xor.
		This gate works for two and three inputs.
	
	Parameters:
		inputs: List of inputs for the gate
	
	Return:
		Output of the logic gate
	'''	
	def LogicXor(self, inputs):
		result = 0
		if len(inputs) == 2:
			orResult   = self.LogicOr(inputs)
			nandResult = self.LogicNand(inputs)
			return self.LogicAnd([orResult,nandResult])
		if len(inputs) == 3:
			norResult1  = self.LogicXor([inputs[0], inputs[1]])
			result  = self.LogicXor([norResult1, inputs[2]])
		return result
	

	
	'''
	Brief: 
		This function runs a logic half adder.
	
	Parameters:
		inputs: List of inputs for the gate
	
	Return:
		List - [Sum, Carry Out]
		
	Example:
		Run([0,1])
		>>> [1,0]
	'''	
	def LogicHalfAdder(self, inputs):
		return [self.LogicXor(inputs), self.LogicAnd(inputs)]
	
	
	
	'''
	Brief:
		This function runs a logic adder
	
	Parameters:
		inputs: List op inputs for the logic gate
				Input order: [A, B, Carry ]
	
	Return:
		List: [Sum, Carry Out]
	
	Example:
		Run([0,1,1])
		>>> [0,1]
		
	'''
	def LogicFullAdder(self, inputs):
		if len(inputs)<3:
			return [0,0]
			
		a     = inputs[0]
		b     = inputs[1]
		carry = inputs[2]
		sum_carry1 = self.LogicHalfAdder([a, b])
		sum_carry2 = self.LogicHalfAdder([sum_carry1[0],carry])
		
		carryOut = self.LogicOr([sum_carry1[1], sum_carry2[1]])
		sum = sum_carry2[0]
		return [sum, carryOut]





















