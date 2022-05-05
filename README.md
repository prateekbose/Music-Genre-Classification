# Music Genre Classification
The objective of the project is to implement a fairly accurate, easy to implement and computationally less taxing model to classify music into separate genres based on various factors.

## Feature Extraction and Techninques used
There are two types of audio features: top-level and low-level. Top-level features give information such as genres, atmosphere, instruments, and so forth. Temporal domain features, frequency domain features, cepstral features, and modulation frequency features are examples of low-level features.

For the purpose of Feature Extraction, Mel-Frequency Cepstral Coefficients (MFCC) where extracted from the training set which were then used in the model.

### Mel-Frequency Cepstral Coefficients (MFCC)
A signal's Mel-Frequency Cepstral Coefficients (MFCC) are a limited group of characteristics (typically 10-20) that simply represent the overall shape of a spectral envelope. To remove the noise, it performs a discrete cosine transform (DCT) on these frequencies. We only maintain a certain sequence of frequencies that have a high possibility of containing information when we use DCT.

### K Nearest Nighbours
It is one of the clustering techniques that collects data points using similarity in proximity strategies. The mean distance between the locations is calculated using Euclidean distance. And the pattern continues to accumulate more points, eventually becoming a cluster. The K-nearest neighbours technique is used in the model as it has consistently shown the best results for this problem in many studies.

## Dataset Used
For the project, the GTZAN dataset was utilised  The GTZAN dataset is the most often used public dataset for music genre recognition assessment in machine learning research (MGR). In order to reflect a diversity of recording settings, the files were acquired in 2000-2001 from a number of sources, including personal CDs, radio, and microphone recordings.

The dataset can be downloaded from: [https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

## Comparitive Results
| Parameters/Methods              | Existing Method 1                   | Existing Method 2    | Propose Method       |
| ------------------------------- | ----------------------------------- | -------------------- | -------------------- |
| Dataset                         | 3-Root and 9-Leaf Genre Dataset     | GTZAN Dataset        | GTZAN Dataset        |
| Train-Test Ratio                | 90% train - 5% validation - 5% test | 90% train - 10% test | 90% train - 10% test |
| Techniques used                 | MIDI with NCD, KNN                  | MFCC, KNN            | MFCC, KNN            |
| Number of DistinguishableGenres | 3                                   | 4                    | 10                   |
| Test Accuracy                   | `85%`                               | `80%`                | `86%`                |