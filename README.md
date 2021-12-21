# Music Genre Classification

## Architecture
Starting from gathering audio tracks, we will use a dataset in which audio tracks will all be in WAV format and their time lengths remain same.
Here I have used k-Nearest Neighbour Algorithm for classification. Through this algorithm we will analyse the MFCC features that are to be acquired through feature extraction. I used the distances between MFCC coefficients in the audio file to recognize a type of music file.

## Results
The average accuracy of the analysis turns out to be 0.86 (86%) for all the tests performed on the sets. With the accuracy being 0.82 (82%) for test sample size of 100 audio samples.

| Test Sample Size      | Random State | Accuracy | 
| ----------- | ----------- |
| 10      | 2       | 1.0 (100%) |
| 10      | 3       | 0.9 (90%) |
| 20      | 58       | 0.9 (90%) |
| 25      | 58       | 0.92 (92%) |
| 100      | 80       | 0.82 (82%) |
