import numpy as np
import typing
import random
from tensorflow.keras.utils import to_categorical
import pickle
def countSamplesClasses(labData: list, numClasses: int) -> np.array:
    countClasses = np.zeros(numClasses)
    for el in labData:
        i = el[1]
        countClasses[int(i)] += 1
    return countClasses

def findIndexes(labData: list, cl: int)-> np.array:
    numSamples = countSamplesClasses(labData, 3)
    ind = np.zeros(int(numSamples[cl]))
    j = 0
    for i in range(len(labData)):
        if labData[i][1] == cl:
            ind[j] = i
            j += 1
    return ind

def findsamples (ind: np.array, num_ind_tr: int, num_ind_test: int) -> typing.Tuple[np.array, np.array]:
    indexes_tr = np.full(num_ind_tr, -1)
    indexes_test = np.full(num_ind_test, -1)
    for i in range(num_ind_tr):
        j = random.randint(0, len(ind) - 1)
        while np.any(indexes_tr == ind[j]):
            j = random.randint(0, len(ind) - 1)
        indexes_tr[i] = ind[j]
    for i in range(num_ind_test):
        j = random.randint(0, len(ind) - 1)
        while np.any(indexes_tr == ind[j]) or np.any(indexes_test == ind[j]):
            j = random.randint(0, len(ind) - 1)
        indexes_test[i] = ind[j]
    return indexes_tr, indexes_test

def concSamples (ind: np.array) -> np.array:
    for i in range(len(ind)):
        if i == 0:
            im = np.load(f'modifieddata/data{int(ind[i])}.npy')
        else:
            im = np.concatenate((im ,  np.load(f'modifieddata/data{int(ind[i])}.npy')))
    return im
def shuffle_arrays(arrays: list, set_seed=-1):
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)

labData = np.load("modifieddata/labDatasets.npy")
ind1 = findIndexes(labData, 0)
ind2 = findIndexes(labData, 1)
ind3 = findIndexes(labData, 2)
ntrainind = 10
ntestind=2
[indexestr1, indexestest1] = findsamples(ind1, ntrainind, ntestind)
print(indexestr1)
print(indexestest1)
[indexestr2, indexestest2] = findsamples(ind2, ntrainind, ntestind)
[indexestr3, indexestest3] = findsamples(ind3, ntrainind, ntestind)
trainind= np.concatenate((indexestr1, indexestr2, indexestr3))
testind= np.concatenate((indexestest1, indexestest2, indexestest3))
imtr1 = concSamples(indexestr1)
imtr2 = concSamples(indexestr2)
imtr3 = concSamples(indexestr3)
labeltr1 = np.zeros(len(imtr1))
labeltr2 = np.ones(len(imtr2))
labeltr3 = np.zeros(len(imtr3))
for i in range(len(imtr3)):
    labeltr3[i] = 2
imtest1 = concSamples(indexestest1)
imtest2 = concSamples(indexestest2)
imtest3 = concSamples(indexestest3)
labeltest1 = np.zeros(len(imtest1))
labeltest2 = np.ones(len(imtest2))
labeltest3 = np.zeros(len(imtest3))
for i in range(len(imtest3)):
    labeltest3[i] = 2
trainX = np.concatenate((imtr1, imtr2, imtr3))
trainY = to_categorical(np.concatenate((labeltr1, labeltr2, labeltr3)))
testX = np.concatenate((imtest1, imtest2, imtest3))
testY = to_categorical(np.concatenate((labeltest1, labeltest2, labeltest3)))
shuffle_arrays([trainX, trainY])
shuffle_arrays([testX, testY])
data = [trainX, trainY, testX, testY]
indexes = [trainind, testind]
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
with open('indexes.pickle', 'wb') as f:
    pickle.dump(indexes, f)

#print(labData)
#numClasses = 3
#count = countSamplesClasses(labData, numClasses)
#print(count)
