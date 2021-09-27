~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Title: Fast Frontal Face and Eye Detection using Viola-Jones Object Detection
Author: Ryan Camilleri (328400L)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- Directory: cascade_500p_1000n_10s
------------------------------------
Contains the face cascade classifier with the following parameters:
# 500 positive images
# 1000 negative images
# 10 training stages
# 0.4 max false alarm rate
# 0.995 min hit rate (default)


- Directory: cascade_500p_1000n_10s_0.991hr
------------------------------------
Contains the face cascade classifier with the following parameters:
# 500 positive images
# 1000 negative images
# 9 training stages
# 0.4 max false alarm rate
# 0.991 min hit rate


- Directory: cascade_500p_1000n_10s_0.999hr
------------------------------------
Contains the face cascade classifier with the following parameters:
# 500 positive images
# 1000 negative images
# 10 training stages
# 0.4 max false alarm rate
# 0.999 min hit rate


- Directory: cascade_1000p_500n_10s
------------------------------------
Contains the face cascade classifier with the following parameters:
# 1000 positive images
# 500 negative images
# 10 training stages
# 0.4 max false alarm rate
# 0.995 min hit rate (default)


- Directory: cascade_750p_750n_10s
------------------------------------
Contains the face cascade classifier with the following parameters:
# 750 positive images
# 750 negative images
# 10 training stages
# 0.4 max false alarm rate
# 0.995 min hit rate (default)


- Directory: haarcascades
------------------------------------
Contains two cascade models for face and eye detection obtained from [1].


- Directory: negatives
------------------------------------
Used to contain the whole data set of negative images used for training [2]. (These were removed due to size restrictions.)


- Directory: positives
------------------------------------
Used to contain the whole data set of positive images used for training [3]. (These were removed due to size restrictions.)


- Directory: test_images
------------------------------------
Contains the 3 test images used for evaluating classifiers in artifact 1.


- File: negatives.txt
------------------------------------
Descriptor file containing paths to all the negative images within the negatives directory


- File: positives.txt
------------------------------------
Descriptor file containing paths to all the positive images within the positives directory. The paths also include the annotated bounding boxes which indicate the face region in the image.


- File: positives.vec
------------------------------------
A vector file created from the positives.txt. This file is required for classifier training. (This was removed due to size restrictions)


- Script: task1.py
------------------------------------
Python script containing code for artifact 1


- Script: task2.py
------------------------------------
Python script containing code for artifact 2


- Script: utils.py
------------------------------------
Python script containing functions to build the negatives.txt and positives.txt files.


Links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[1] https://github.com/opencv/opencv/tree/master/data/haarcascades
[2] https://www.kaggle.com/yeayates21/garage-detection-unofficial-ssl-challenge
[3] https://www.kaggle.com/greatgamedota/ffhq-face-data-set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~