import numpy as np
import os
import pickle
import operator
from sklearn.model_selection import train_test_split  
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from python_speech_features import mfcc
import scipy.io.wavfile as wav

directory = os.getcwd()

def distance(inst1, inst2, k):
    X1 = inst1[0]
    Y1 = inst1[1]
    X2 = inst2[0]
    Y2 = inst2[1]
    dist = np.trace(np.dot(np.linalg.inv(Y2), Y1)) 
    dist +=(np.dot(np.dot((X2-X1).transpose() , np.linalg.inv(Y2)) , X2-X1 )) 
    dist += np.log(np.linalg.det(Y2)) - np.log(np.linalg.det(Y1))
    dist -= k
    return dist

def getNeighbors(train_data, instance, k):
    distances = []
    for data in train_data:
        dist = distance(data, instance, k) + distance(instance, data, k)
        distances.append((data[2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for neighbor in range(k):
        neighbors.append(distances[neighbor][0])
    return neighbors

def nearestNeighbor(neighbors, result):
    votes = {}
    for neighbor in neighbors:
        if neighbor in votes:
            votes[neighbor]+=1
        else:
            votes[neighbor] = 1
    sorted_votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    #print(result, end=': {')
    #for neighbor in sorted_votes:
    #    print(results[neighbor[0]-1], ':', neighbor[1],end=', ')
    #print("}\n")
    return sorted_votes[0][0]

# def accuracy(test_data, pred):
#     count = 0
#     for i in range(len(test_data)):
#         if(test_data[i] == pred[i]):
#             count+=1
#     return count

def loadDataset(filename, size, split):
    dataset = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                dataset.append(pickle.load(fr))
        except EOFError:
            pass
    train_set, test_set= train_test_split(dataset, test_size= size, random_state=split)  
    return train_set, test_set

def predict_unknown(file, res, samples_results, samples_corr):
    train, test = loadDataset(os.getcwd() + "/features.dmp", 1, 58)
    (rate,sig)=wav.read(file)
    mfcc_feat = mfcc(sig, rate, winlen=0.02, winstep=0.01, numcep=15,nfft = 1200, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix,covariance,0)
    pred = nearestNeighbor(getNeighbors(train, feature, 10), res)
    samples_results.append(results[pred-1])
    if res == results[pred-1]:
        samples_corr += 1
    return samples_results, samples_corr

results = []

for folder in os.listdir("./features/"):
    results.append(folder)

accuracy_count = []
leng_count = []
a = 57
b = 63
pass_count = 3
highest_acc = 0
sizes = [0.01, 0.01, 0.02, 0.025, 0.1]
state = [2, 3, 58, 58, 80]


print(" ------ Testing Testing Samples --------")
for i in range(len(state)):
    #print("a: %d, b: %d" % (a, b))
    # random_state = random.randint(a,b)
    print("test_size: %d, random_state: %d" % (sizes[i]*1000, state[i]))
    train, test = loadDataset(os.getcwd() + "/features.dmp", sizes[i], state[i])
    predictions = []
    test_pred = []
    leng = len(test)
    for x in tqdm(range(leng)):
        neighbor = nearestNeighbor(getNeighbors(train, test[x] , 10), results[test[x][-1]-1])
        # print("%s: %s" % (results[test[x][-1]-1], results[neighbor-1]))
        #print(results[test[x][-1]-1], end=': ')
        test_pred.append(results[test[x][-1]-1]) 
        predictions.append(results[neighbor-1]) 
    accuracy_count.append(accuracy_score(test_pred, predictions, normalize=False))
    leng_count.append(leng)
    if(i == 4):
        print(confusion_matrix(test_pred, predictions))
    print(f1_score(test_pred, predictions, average='weighted'))
    print(precision_score(test_pred, predictions, average='weighted', zero_division=1))
    # if(i >= 0 and accuracy_count[i] > highest_acc):
    #     highest_acc = accuracy_count[i]
    #     if(abs(a-random_state) > abs(b-random_state)):
    #         b = random_state
    #     else:
    #         a = random_state
    print("Pass %d: %0.2f%c accuracy" % (i+1, 100*accuracy_score(test_pred, predictions, normalize=False)/leng, '%'))

print("Average accuracy: %0.2f%c" % (100*sum(accuracy_count)/sum(leng_count), '%'))

print(" ------ Testing Unknown Samples --------")

samples = ["jazz", "pop", "rock", "country", "metal"]
samples_results = []
samples_corr = 0
for i in tqdm(range(len(samples))):
    location = os.getcwd() + "/" + samples[i] + "_sample.wav"
    samples_results, samples_corr = predict_unknown(location, samples[i], samples_results, samples_corr)

for i in range(len(samples)):
    print("%s: %s" % (samples[i], samples_results[i]))

print("Correct predictions count: %d" % samples_corr)