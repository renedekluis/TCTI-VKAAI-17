import random


'''
********************************************************
Pseudo code van Wikipedia:
	Kies initiële populatie
	Bepaal voor elk individu de fitness
	Herhaal tot de stop-conditie vervuld is:
		Selecteer de individuen uit de huidige populatie
		Reproductie beste individuen
		Bepaal voor elk individu de fitness
********************************************************
'''



'''
	@brief Create an individual
	
	@param[in]	length 	Length of individual characteristics
	@param[in]	min		Minimum value of characteristic
	@param[in]	max		Maximum value of characteristic
	
	@return	One individual
'''
def createIndividual(length, min, max):
	individuals = []
	for x in range(length):
		individuals.append(random.randint(min, max))
		
	return individuals



'''
	@brief Create a population
	
	@param[in]	amountIndividuals 	Amount of individuals in the population
	@param[in]	length 				Length of individual characteristics
	@param[in]	min					Minimum value of characteristic
	@param[in]	max					Maximum value of characteristic
	
	@return	A population
'''
def createPopulation(amountIndividuals, length, min, max):
	population = []
	for x in range(amountIndividuals):
		population.append(createIndividual(length, min, max))
	return population




'''
	@brief Sort the population
	
	@param[in]	population 	the population
	@param[in]	expectedSummed 		Expected value of all values summed
	@param[in]	expectedMultiplied	Expected value of all values multiplied
	
	@return	Sorted Population
'''
def sortGeneration(population, expectedSummed, expectedMultiplied):
	fitnessList = []
	for individual in population:
		fitnessList.append(fitness(individual, expectedSummed, expectedMultiplied))
	
	individuals = []
	for fit,individual in sorted(zip(fitnessList,population)):
		individuals.append(individual)
	return individuals
	



'''
	@brief Create a new population from the best individuals
	
	@param[in]	population 		The population
	@param[in]	keepBestAmount 	Ammount of Individuals to keep
	
	@return	New Population
'''	
def createNewGeneration(population, keepBestAmount):
	for individual in population[keepBestAmount:]:
		index = random.randint(0, len(individual)-1)
		individual[index] = ( 1 if individual[index] == 0 else  0 )
	return population


'''
	@brief Determine the fitness of an individual
	
	@param[in]	individual 			individual
	@param[in]	expectedSum 		Expected summed value
	@param[in]	expectedMultiply 	Expected multiplied value
	
	@return	Fitness
'''	
def fitness(individual, expectedSum, expectedMultiply):
	sum 		= 0
	multiply 	= 0
	for index in range(len(individual)):
		if individual[index] == 0:
			sum += index + 1
		else:
			multiply = ( index + 1 if multiply == 0 else multiply * (index + 1))
	return abs(sum - expectedSum)+abs(multiply - expectedMultiply)



'''
	@brief Check if all listed values are the same
	
	@param[in]	itemList 			list with items
	
	@return	boolean if all items are the same
'''	
def allSame(itemList):
	return all(x == itemList[0] for x in itemList)












#===================================================================================================
# MAIN PROGRAM
#===================================================================================================

expectedSum 			= 36	# Expected summed value
expectedMultiply 		= 360	# Expected multiplied value
maxGenerationsAmount 	= 100	# Maximum generations amount
FitnessList 			= []	# List to keep track of past fitness results
FitnessCheckAmmount 	= 15	# Times the fitness has to be the same to be called 'Best'
population 				= createPopulation(100, 10, 0, 1) # The population

for x in range(FitnessCheckAmmount):
	FitnessList.append(99)

cnt = 0
for generation in range(maxGenerationsAmount):
	cnt+=1
	population = sortGeneration(population, expectedSum, expectedMultiply)
	population = createNewGeneration(population, 10)
	f = fitness(population[0], expectedSum, expectedMultiply)
	FitnessList.append(f)
	if len(FitnessList) > FitnessCheckAmmount:
		FitnessList = FitnessList[1:]
	if allSame(FitnessList):
		print('Best answer:        ', population[0])
		print('Fitness:            ', FitnessList[-1] )
		print('Generations needed: ',cnt)
		break
	#print(FitnessList)

