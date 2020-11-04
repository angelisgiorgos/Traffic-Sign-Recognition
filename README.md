# **Traffic-Sign-Recognition**
Traffic Sign recognition using Convolutional Neural Networks.

Usage steps:
* To use it in your local machine:
  * Clone the repo
  * Download the German Traffic Signs Dataset.
  * Navigate to traffic_sign_classification folder.
  * Open main.py
  * Assign to in_dir, data_dir the correct directories.
  * Run main.py
  
* To use it in Google Colab, open the .ipynb file and run all the cells.

Used German Traffic Sign Recognition Benchmark. The whole dataset consists of 50.000 traffic sign images. Approximately the 39.000 are used for training and validation
and 12.000 for testing.
> Dataset is provided here: http://benchmark.ini.rub.de/?section=gtsrb&subsection=news

Below is shown the distribution of traffic signs between classes:
* Blue color: Training set.
* Orange color: Validation set.
* Green color: Testing set

<p align="center">
  <img src="https://github.com/georange7/Traffic-Sign-Recognition/blob/master/png/samples.png">
</p>

A small overview of dataset's images:
<p align="center">
  <img src="https://github.com/georange7/Traffic-Sign-Recognition/blob/master/png/random_training_Set.png">
</p>

We trained the GTSRB in three different CNN architectures. The first one was LeNet-5, the second one was a custom architecture, the thrid one MicronNet one of the most accurate
CNN architectures in traffic sign classification.

Architecure | Accuracy | Precision | Recall | F1 score 
------------ | ------------- | ------------- | ------------- | -------------
**LeNet-5** | 0.87 | 0.81 | 0.79 | 0.79
**Custom Model** | 0.94 | 0.93 | 0.92 | 0.92
**MicronNet** | 0.96 | 0.94 | 0.93 | 0.94


### Recognition results:
<p align="center">
  <img src="https://github.com/georange7/Traffic-Sign-Recognition/blob/master/png/recognition.png">
</p>
