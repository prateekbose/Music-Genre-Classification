import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import pickle

directory = os.getcwd() + "/features"
f = open("features.dmp", "wb")
i = 0

for folder in os.listdir(directory):
    print(folder, end=' ')
    i+=1
    if i == 11: 
        break
    for file in os.listdir(directory+"/"+folder):
        (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
        mfcc_feat = mfcc(sig, rate, winlen=0.02, winstep=0.01, numcep=15,nfft = 1200, appendEnergy=False)   
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix , covariance , i)
        pickle.dump(feature , f)
    print("--- done ({}/10)".format(i))
f.close()
print("---- feature dump finished ----")