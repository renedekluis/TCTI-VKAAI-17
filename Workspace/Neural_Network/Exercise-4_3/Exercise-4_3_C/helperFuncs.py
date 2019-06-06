
import math

'''
\brief Get G'(in) function
'''
def g( input ):
	return 1-math.tanh(math.tanh())



'''
\brief Get Activation Error
'''
def activationError(y, a):
	return (y-a)


'''
\brief Get deltaerror for Output Node
'''
def deltaK(input, desiredOutput, actualOutput):
	return g(input)*activationError(desiredOutput-actualOutput)


def sumDeltaJ():
	return 




def deltaJ(weight, delta):
	return weight * delta()


def activate( total ):
	return math.tanh( total )



def BackProb(currentWeight, learnRate, activation, delta):
	





