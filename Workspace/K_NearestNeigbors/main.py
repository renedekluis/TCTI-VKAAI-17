from k_nearest_neighbors import *


dataset1_csv = "../datasets/dataset1.csv"
validation1_csv = "../datasets/validation1.csv"
days_csv = "../datasets/days.csv"

conv = {5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)}
testSet = np.genfromtxt(days_csv, delimiter=';', usecols=[1,2,3,4,5,6,7], converters=conv)
validationSet = np.genfromtxt(validation1_csv, delimiter=';', usecols=[1,2,3,4,5,6,7], converters=conv)



knn = KNearestNeighbours(dataset1_csv, 1)


validationDates = np.genfromtxt(validation1_csv, delimiter=';', usecols=[0])

validationLabels = []


for label in validationDates:
	if label < 20010301:
		validationLabels.append('winter')
	elif label >= 20010301 and label < 20010601:
		validationLabels.append('lente')
	elif label >= 20010601 and label < 20010901:
		validationLabels.append('zomer')
	elif label >= 20010901 and label < 20011201:
		validationLabels.append('herfst')
	else: # from 01ô€€€12 to end of year
		validationLabels.append('winter')



for k in range(40,70):
	
	result = knn.Run(validationSet, k)
	correct = 0
	for idx, x in enumerate(validationLabels):
		if x == result[idx]:
			correct +=1
	
	print(k,' - ', correct,'%  ')

	knn.k = k

	
	
print(knn.Run(testSet, 59))
'''
Beste K: 59
'''

























