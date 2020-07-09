#Random Forest Classifier of Heartbeats
#Heartbeat Sounds Source: https://www.kaggle.com/kinguistics/heartbeat-sounds/data?select=set_a_timing.csv
#Script takes in .wav files, extracts features of sound files into tabular data
#Tabular data is used to train the random forest classifier
#Best Metrics:
#To Do: - Include more Features in pre-processing
#       - Adjust hyperparameters

import numpy as np 
import pandas as pd 
import os,fnmatch
import scipy
import librosa
import IPython.display as ipd

# Data Pre-Processing - Creating Tabular Data
def heart_df(set_files,col_names,d_list): #Creates tabular data of sound file features for each sound file
    output =[]
    count =0
    for folder in set_files:
        for d in d_list: #list of outcomes
            values= fnmatch.filter(os.listdir(folder),d)
            label= d.split("*")[0]
            
            for value in values: #calculating and appending sound file features from librosa
                x,sr=librosa.load(str(folder +'\\'+ value),duration=5,res_type='kaiser_fast')
                output.append([np.mean(x) for x in librosa.feature.mfcc(x,sr=sr)])
                output[count].append(sum(librosa.zero_crossings(x)))
                output[count].append(np.mean(librosa.feature.spectral_centroid(x)))
                output[count].append(np.mean(librosa.feature.spectral_rolloff(x,sr=sr)))
                output[count].append(np.mean(librosa.feature.chroma_stft(x,sr=sr)))
                output[count].append(label)
                output[count].append(value)
                count+=1
    return pd.DataFrame(output,columns=col_names)

music_folders=["set_a","set_b"]

col_names =["mfkk"+str(i) for i in range(20)]
for i in ["zero","centroid","rolloff","chroma","outcome","file"]:
    col_names.append(i)
    
print("Heart Studio, Start!")
print("Extracting Features from sound files ... this may take a while.")    
outcomes=["normal*.wav","artifact*.wav","murmur*.wav"] 

feature_df= heart_df(music_folders,col_names,outcomes)
print ("\nPreprocessing Information --------------------")
print ("Shape: " + str(feature_df.shape))
print("\nOutcome      Counts" )
print(feature_df["outcome"].value_counts())

# Splitting Data into Training and Test Sets
X = feature_df.iloc[:,0:24]
y=feature_df["outcome"]

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder().fit(y)
y=l.transform(y)

from sklearn.model_selection import train_test_split,GridSearchCV
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=31)
print("\nX Train: ",len(X_train),"\n","X Test: ",len(X_test),sep="")

# Training the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(max_depth= 8,
 max_features= 5,
 min_samples_split=5,
 n_estimators=500, random_state=31).fit(X_train,y_train)
forest

# Outputting Final Results
print("\nFinal Results --------------------")
from sklearn.metrics import accuracy_score
y_pred=forest.predict(X_test)
print("Accuracy:" + str(accuracy_score(y_test,y_pred)))