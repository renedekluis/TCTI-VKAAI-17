from logic_gates import *

twoInputOptions = [
	[0,0],
	[1,0],
	[0,1],
	[1,1] 
]

threeInputOptions = [
	[0,0,0],
	[0,0,1],
	[0,1,0],
	[0,1,1],
	[1,0,0],
	[1,0,1],
	[1,1,0],
	[1,1,1]
]


	
logicGates = LogicGates()

logicGateFunctions = [
	("NOR",  logicGates.LogicNor ),
	("OR",   logicGates.LogicOr  ),
	("AND",  logicGates.LogicAnd ),
	("NAND", logicGates.LogicNand),
	("XOR",  logicGates.LogicXor ),
	("Half Adder",  logicGates.LogicHalfAdder )
]




for gate in logicGateFunctions:
	print('\n',gate[0],'\n------')
	for option in twoInputOptions:
		print(option, gate[1](option))




print('\n Full Adder\n------')
for option in threeInputOptions:
	print(option, logicGates.LogicFullAdder(option))

