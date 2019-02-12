from neuron import *


class LogicGates:
	def __init__(self):
		self.norGate  = Neuron([-0.5, -0.5],  0  )
		self.orGate   = Neuron([ 0.5,  0.5],  0.5)
		self.andGate  = Neuron([ 0.5,  0.5],  1  )
		self.nandGate = Neuron([-0.5, -0.5], -0.5)



	def LogicNor(self, inputs):
		return self.norGate.Run(inputs)


	def LogicOr(self, inputs):
		return self.orGate.Run(inputs)


	def LogicAnd(self, inputs):
		return self.andGate.Run(inputs)


	def LogicNand(self, inputs):
		return self.nandGate.Run(inputs)
		
	def LogicXor(self, inputs):
		orResult   = self.LogicOr(inputs)
		nandResult = self.LogicNand(inputs)
		return self.LogicAnd([orResult,nandResult])
	

	
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
		sum_carry1 = self.LogicHalfAdder([a,b])
		sum_carry2 = self.LogicHalfAdder([sum_carry1[0],carry])
		
		carryOut = self.LogicOr([sum_carry1[1], sum_carry2[1]])
		sum = sum_carry2[0]
		return [sum, carryOut]





















