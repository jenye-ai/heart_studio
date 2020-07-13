# heart_studio

## Files
### hr_analysis.ipynb
Takes in a .wav file of a heart auscultation and analyzes the signal to find bpm (beats per minute).

### rf_classifier.ipynb/.py
Trains a random forest classifier for classification of types of heart signals (normal, murmur, artifact) given a set of .wave files.

## Reproducing Results
Recommended IDE: Jupyter Notebook

### Download Required Data Files:
Please Download Folders "set_a" and "set_b" from: https://www.kaggle.com/kinguistics/heartbeat-sounds/data?select=set_a_timing.csv

### Create a new virtual environment
In your desired folder, type the following in your command line:
>python3 -m venv [ENV_NAME]

>source [ENV_NAME]/bin/activate

please cd into project folder before running any script

### Requirements
Please install all dependences from the requirements.txt file by running the following in your command line:
> pip install -r requirements.txt.

