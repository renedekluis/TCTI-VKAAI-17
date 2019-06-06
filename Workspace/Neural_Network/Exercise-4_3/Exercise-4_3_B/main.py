from neuron import *

print('/'*100)
print('/'*100)
print('/'*100)

orOptions = [
	[0,0,0],
	[1,0,1],
	[0,1,1],
	[1,1,1] 
]

norOptions = [
	[0,0,1],
	[1,0,0],
	[0,1,0],
	[1,1,0] 
]
andOptions = [
	[0,0,0],
	[1,0,0],
	[0,1,0],
	[1,1,1] 
]
nandOptions = [
	[0,0,1],
	[1,0,1],
	[0,1,1],
	[1,1,0] 
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


logicOr   = Neuron('LOGIC_OR', 2, 0.01)
logicNor  = Neuron('LOGIC_NOR', 2, 0.01)
logicAnd  = Neuron('LOGIC_AND', 2, 0.01)
logicNand = Neuron('LOGIC_NAND', 2, 0.01)

trainRange = 10000

for idx in range(1,len(orOptions)):
	for x in range(trainRange):
		logicOr.Update([orOptions[idx][0],orOptions[idx][1]], orOptions[idx][2])

for idx in range(1,len(norOptions)):
	for x in range(trainRange):
		logicNor.Update([norOptions[idx][0],norOptions[idx][1]], norOptions[idx][2])

for idx in range(1,len(andOptions)):
	for x in range(trainRange):
		logicAnd.Update([andOptions[idx][0],andOptions[idx][1]], andOptions[idx][2])

for idx in range(1,len(nandOptions)):
	for x in range(trainRange):
		logicNand.Update([nandOptions[idx][0],nandOptions[idx][1]], nandOptions[idx][2])

		

print('\n OR GATE\n-------')
for input in orOptions:
	print([input[0], input[1]], logicOr.Run([input[0], input[1]]))
	
print('\n NOR GATE\n-------')
for input in norOptions:
	print([input[0], input[1]], logicNor.Run([input[0], input[1]]))
	
print('\n AND GATE\n-------')
for input in andOptions:
	print([input[0], input[1]], logicAnd.Run([input[0], input[1]]))
	
print('\n NAND GATE\n-------')
for input in nandOptions:
	print([input[0], input[1]], logicNand.Run([input[0], input[1]]))






