from neural_network import *


# OR
train1 = [
	[0,0,0],
	[1,0,1],
	[0,1,1],
	[1,1,1],
]

# NOR
train2 = [
	[0,0,1],
	[1,0,0],
	[0,1,0],
	[1,1,0],
]
# XOR
train3 = [
	[0,0,0],
	[1,0,1],
	[0,1,1],
	[1,1,0],
]
# AND
train4 = [
	[0,0,0],
	[1,0,0],
	[0,1,0],
	[1,1,1],
]
# NAND
train5 = [
	[0,0,1],
	[1,0,1],
	[0,1,1],
	[1,1,0],
]


testSet = [
	[0,0],
	[1,0],
	[0,1],
	[1,1],
]

NN_Or 	= NeuralNetwork('Logic Or', 	2, [2,2,1], 	1, 0.001)
NN_Nor 	= NeuralNetwork('Logic Nor', 	2, [2,2,1], 	1, 0.001)
NN_Xor 	= NeuralNetwork('Logic Xor', 	2, [2,3,2], 1, 0.0001)
NN_And 	= NeuralNetwork('Logic And', 	2, [2,2], 	1, 0.0001)
NN_Nand = NeuralNetwork('Logic Nand', 	2, [2,3], 	1, 0.0001)



print('Training Or')
trainDuration = 10000
for i in range(trainDuration):
	for option in train1:
		#print('OR: ',i,'/',trainDuration,[option[0],option[1]],[option[2]], '-->', NN_Or.FeedForward([option[0],option[1]]))
		NN_Or.Train([option[0],option[1]],[option[2]])
		

print('Training Nor')
trainDuration = 10000
for i in range(trainDuration):
	for option in train2:
		#print('NOR: ',i,'/',trainDuration,[option[0],option[1]],[option[2]], '-->', NN_Nor.FeedForward([option[0],option[1]]))
		NN_Nor.Train([option[0],option[1]],[option[2]])

print('Training Xor')
trainDuration = 10000
for i in range(trainDuration):
	for option in train3:
		#print('XOR: ',i,'/',trainDuration,[option[0],option[1]],[option[2]], '-->', NN_Xor.FeedForward([option[0],option[1]]))
		NN_Xor.Train([option[0],option[1]],[option[2]])


print('Training AND')
trainDuration = 10000
for i in range(trainDuration):
	for option in train4:
		#print('AND: ',i,'/',trainDuration,[option[0],option[1]],[option[2]], '-->', NN_And.FeedForward([option[0],option[1]]))
		NN_And.Train([option[0],option[1]],[option[2]])

print('Training NAND')
trainDuration = 10000
for i in range(trainDuration):
	for option in train5:	
		#print('NAND: ',i,'/',trainDuration,[option[0],option[1]],[option[2]], '-->', NN_Nand.FeedForward([option[0],option[1]]))
		NN_Nand.Train([option[0],option[1]],[option[2]])	


print('\n'*15)

print('\n'*5,'Network: Or')
print('-----'*5)
for option in train1:
	op = [option[0],option[1]]
	result = NN_Or.FeedForward(op)[0]
	rounded = round(result,3)
	print(op, option[2],'\t',rounded,'--> (',result,')')
	
	
print('\n'*5,'Network: Nor')
print('-----'*5)
for option in train2:
	op = [option[0],option[1]]
	result = NN_Nor.FeedForward(op)[0]
	rounded = round(result,3)
	print(op, option[2],'\t',rounded,'--> (',result,')')
	
	
print('\n'*5,'Network: Xor')
print('-----'*5)
for option in train3:
	op = [option[0],option[1]]
	result = NN_Xor.FeedForward(op)[0]
	rounded = round(result,3)
	print(op, option[2],'\t',rounded,'--> (',result,')')
	

print('\n'*5,'Network: And')
print('-----'*5)
for option in train4:
	op = [option[0],option[1]]
	result = NN_And.FeedForward(op)[0]
	rounded = round(result,3)
	print(op, option[2],'\t',rounded,'--> (',result,')')
	

print('\n'*5,'Network: Nand')
print('-----'*5)
for option in train5:
	op = [option[0],option[1]]
	result = NN_Nand.FeedForward(op)[0]
	rounded = round(result,3)
	print(op, option[2],'\t',rounded,'--> (',result,')')












