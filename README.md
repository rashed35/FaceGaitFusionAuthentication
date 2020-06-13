# Person Identification using Face and Gait Fusion
This was an assignment in one of my courses in the last semester (Spring 2020). From the problem statement: 
> The goal of this Assignment is to use Principal Components Analysis (PCA), K-Nearest
Neighbours (kNN) and Random Forest to recognize face images and gait signals.

I experimented with some more features and classifiers along with the ones mentioned in our problem description. Find the detailed report that I prepared for the assignment in [FinalReport.pdf](FinalReport.pdf). 

## Dataset
I will not publish the data given to us as I don't have the permission to do so. However, I am sharing the codes and the results.

1. **Face Dataset**: Face images of 10 people, with each person captured under 24 different lighting conditions, for a total of 240 images. These face images taken from the CMU PIE database.
2. **Gait Dataset**: This is a gait dataset collected using Inertial Measurement Unit (IMU) sensors, with accelerometer data for the axes x, y and z. For each person, the dataset provides a csv file of acceleration values for x, y, z read at a frequency of 100Hz. These gait signals collected by NUS researchers.

## Facial Features
1. PCA: Principle Component Analysis. 
2. LBP: Local Binary Pattern. 
3. SIFT: Scale-Invariant Feature Transform.
4. SURF: Speeded Up Robust Features.
5. CNN: Convolutional Neural Network.

## Gait Features
1. Statistical Features (e.g. Mean, Std. deviation, Variance, etc.)
2. LSTM Features.

## Classifiers for each (Face & Gait)
1. kNN: k-Nearest Neighbors.
2. Random Forest.
3. Support Vector Machine.

## Fusion of Classifiers
The fusion is a score-based fusion. The parameter (&alpha;) controls how much importance to give to each of the scores. From the problem statement:
> The Final prediction from fusion is calculated by using the following formula,
> <img src="https://latex.codecogs.com/svg.latex?Pred%20=%20\alpha%20*%20face%20prediction%20+%20(1%20-%20\alpha)%20*%20gait%20prediction"></img>