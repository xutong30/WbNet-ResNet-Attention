# WbNet-ResNet-Attention

Datasets:

WINGBEATS: https://www.kaggle.com/potamitis/wingbeats

ABUZZ: https://web.stanford.edu/group/prakash-lab/cgi-bin/mosquitofreq/the-science/data/

# DataGeneration.py:
Generating training data, validation data

Generating .csv files:

All csv files formatting: [data_name, genera, species]

Each dataset generates 3 .csv files, data will be generated randomly each time to run this python file.

.csv 1: all data

.csv 2: listing training data information

.csv 3: listing validation data information


# Data_prepro.py:
Data preprocessing: convert raw audio into mel-spectrograms

And load data into data loader, so to use in Train.py.


# Model.py:

Our Model: combines ResNet-18 and the Self-Attention Mechanism



# Train.py:

Training the model and validating the model


# Code running procedure:
Download 2 datasets.
For Wingbeats dataset: unzip the dataset, and it can be used directly to generate .csv files in DataGeneration.py

For the Abuzz dataset: since it has more mosquito species files than we need, we will only use six species. Firstly, create a folder called Abuzz_raw, and then make 6 sub-folders called -  Ae. aegypti, Ae. albopictus, An. arabiensis, An. gambiae, C. pipiens, and C. quinquefasciatus. Move all the audios to their corresponding subfolders from the downloaded dataset. Remember to remove the background audios - ie. some audios named background.wav or contained the letters ‘bg’ in their audios names. And then clips each audio in segments, and each segment remains the same length. In our paper, each audio remains for 10 secs. The processed dataset has been uploaded to GitHub for convenience. The uploaded dataset can be used directly to generate .csv files in DataGeneration.py. Please note that the .csv files of 2 datasets will be generated sequentially, please comment (ctrl+/) another one if you only want to generate .csv files for one of the datasets.

All sample .csv files are uploaded to GitHub. 

Run Train.py to start training our model. All parameters can be changed in the __main__ function. 
