import math


inputSum = [0.7615941559557649, 0.7615941559557649, 1]
tot =0
for x in inputSum:
	tot+=x*0.5
print('calculated input = ', tot)
	
	
input =  tot

print('Tanh(',input,') =',math.tanh(input))

print('g`(',input,') = ',(1- math.tanh(math.tanh(input))))


