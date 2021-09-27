import cv2 as cv
import os

# Step 1: Data Retrieval
# Positive dataset found consisted of 70k image thumbnails of faces
# Positives data set: https://www.kaggle.com/greatgamedota/ffhq-face-data-set - 70 000 positives
# Negative dataset found consisted of 3174 images of garages which do not contain faces
# Negatives data set: https://www.kaggle.com/yeayates21/garage-detection-unofficial-ssl-challenge - 3174 negatives
#
# Step 2: Data Engineering
# For training, descriptor files are required for both negative and positive samples.
# --> The negative descriptor requires the negative image path.
# --> The position descriptor requires the positive image path and bounding box (BB) coordinates of faces within the images.
# ----> To avoid the lengthy process of manual annotation the psotives dataset was found specifically to include 1 face per image.
# ----> Since thumbnails (128x128) were used, the face in each image fills a good portion of it. Therefore, the BB coordinates was set to encapsulate the whole thumbnail.
# To create the above descriptors, the utils.py script was defined.
#
# Classifier Training also requires a vector file of positive samples to be created from the positives descriptor defined above
# To do this OpenCv tools were installed
# The following command was then run to generate the vector file: "positives.vec"
# --> Command: C:\Users\Owner\opencv\build\x64\vc15\bin/opencv_createsamples.exe -info positives.txt -w 24 -h 24 -num 70000 -vec positives.vec
# ----> The chosen window size is 24 by 24 similar to that used in the Viola-Jones paper. Larger windows correspond to longer training times but more accuracy. However, 24 x 24 is a good compromise.
# ----> Since every image consists of 1 BB, the number of BBs to be considered was set equal to the total number of positive images
#
# Step 3: Model Training
# For this project, training had to be done to determine the effect of varied image ratios and minimum hit rates.
# Therefore, 3 initial directories were created to hold the trained models with 1:2, 1:1 and 2:1 image ratios respectively for positive and negative images.
# After determining the victor from the 3 models, 2 further directories were created to store the models with the superior image ratio and varied minimum hit rates.
#
# Time to train our model.
# Hyperparameters:
# --> Window Size: This must have the same width and height as those used to create the vector file in Step 2.
# --> Number of stages: The more stages the classifier has, the better the accuracy. However, by having a very deep model, this may be overfit to training examples and so, a balance must be found.
# --> Number of positives: Number of positive images to use when training
# --> Number of negatives: Number of negative images to use when training
# --> Max False Alarm Rate: The max probability of falsely rejecting the null hypothesis for a particular test (0.5 by default)
# --> Minimum hit rate : The larger the minimum rate, the more features are required for the model to reach its goals. Essentially, with higher rates, the model is forced to overfit to the training data. If too low however, the model would not develop features that resemble faces well. (0.995 by default)
#
# The following command trains a model on a 1:2 image ratio with 10 stages, 0.4 max false alarm and 0.991 min hit rate
# Command: C:\Users\Owner\opencv\build\x64\vc15\bin/opencv_traincascade.exe -data cascade_500p_1000n_10s_0.991hr/ -vec positives.vec -bg negatives.txt -w 24 -h 24 -numPos 500 -numNeg 1000 -numStages 10 -maxFalseAlarmRate 0.4 -minHitRate 0.991
#
#
# Step 4: Model Observations
# Initially, the maximum number of stages was set out to be 15. However, it was observed that during the training process, the acceptance ratio for negative images was exceedingly low which would result in over fitting. To account for this, the number of stages was set out to be 10
# FA rate was made to 0.4 instead of the default 0.5 which forces individual stages to be more complex. This is suggested if you dont want to add more data.
#
# - Training times were as follows:
# -- Model with pos 500 and neg 1000 = 3min 50s
# -- Model with pos 1000 and neg 500 = 4min 5s
# -- Model with pos 750 and neg 750 = 3min 42s
#
# After applying the models using task1.py and comparing the precision, recall and f-measure of each model (see report), it is determind that the model with 500 positives and 1000 negatives is superior.
#
# To determine if the superior model could be enhanced by varying hit rates, 2 models were trained:
# -- Model with pos 500, neg 1000 and minHR 0.999 = 4mins 27s
# ----> Since this model had a higher hit rate, it was observed to include a lot of features especially in later stages.
# -- Model with pos 500, neg 1000 and minHR 0.991 = 2mins 33s
# ----> Since this model was less strict due to having a lower min hit rate, the required false alarm rate was reached by stage 9 instead of the max 10.


def load_test_images():
    images = []
    folder = "test_images"

    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

    return images


def draw_rectangles(image, rectangles):
    # these colors are actually BGR
    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    modified_image = image.copy()

    for (x, y, w, h) in rectangles:
        # determine the box positions
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        # draw the box
        cv.rectangle(modified_image, top_left, bottom_right, line_color, lineType=line_type)

    return modified_image


# load test images
test_images = load_test_images()

# load the trained models with varied image ratios
cascade_500p_1000n_10s = cv.CascadeClassifier('cascade_500p_1000n_10s/cascade.xml')
cascade_750p_750n_10s = cv.CascadeClassifier('cascade_750p_750n_10s/cascade.xml')
cascade_1000p_500n_10s = cv.CascadeClassifier('cascade_1000p_500n_10s/cascade.xml')

# load trained models using the best image ratio but varied hit rates
cascade_500p_1000n_10s_high_hr = cv.CascadeClassifier('cascade_500p_1000n_10s_0.999hr/cascade.xml')
cascade_500p_1000n_10s_low_hr = cv.CascadeClassifier('cascade_500p_1000n_10s_0.991hr/cascade.xml')

print("What do you want to evaluate?")
print("----------------------------------------------")
print("(1) Cascades with varied image ratios")
print("(2) Cascades with varied minimum hit rates")
print("----------------------------------------------")
evaluationStep = input("Input integer choice: ")

if evaluationStep == "1":
    for test_image in test_images:

        # Get detected regions
        cascade_500p_1000n_10s_regions = cascade_500p_1000n_10s.detectMultiScale(test_image)
        cascade_750p_750n_10s_regions = cascade_750p_750n_10s.detectMultiScale(test_image)
        cascade_1000p_500n_10s_regions = cascade_1000p_500n_10s.detectMultiScale(test_image)

        # draw the detection results onto the original image
        cascade_500p_1000n_10s_image = draw_rectangles(test_image, cascade_500p_1000n_10s_regions)
        cascade_750p_750n_10s_image = draw_rectangles(test_image, cascade_750p_750n_10s_regions)
        cascade_1000p_500n_10s_image = draw_rectangles(test_image, cascade_1000p_500n_10s_regions)

        # display the images
        cv.imshow('cascade_500p_1000n_10s_image', cascade_500p_1000n_10s_image)
        cv.imshow('cascade_750p_750n_10s_image', cascade_750p_750n_10s_image)
        cv.imshow('cascade_1000p_500n_10s_image', cascade_1000p_500n_10s_image)

        cv.waitKey(0)

elif evaluationStep == "2":
    for test_image in test_images:
        # Get detected regions
        cascade_500p_1000n_10s_regions = cascade_500p_1000n_10s.detectMultiScale(test_image)
        cascade_500p_1000n_10s_high_hr_regions = cascade_500p_1000n_10s_high_hr.detectMultiScale(test_image)
        cascade_500p_1000n_10s_low_hr_regions = cascade_500p_1000n_10s_low_hr.detectMultiScale(test_image)

        # draw the detection results onto the original image
        cascade_500p_1000n_10s_image = draw_rectangles(test_image, cascade_500p_1000n_10s_regions)
        cascade_500p_1000n_10s_high_hr_image = draw_rectangles(test_image, cascade_500p_1000n_10s_high_hr_regions)
        cascade_500p_1000n_10s_low_hr_image = draw_rectangles(test_image, cascade_500p_1000n_10s_low_hr_regions)

        # display the images
        cv.imshow('cascade_500p_1000n_10s_default_hr_image', cascade_500p_1000n_10s_image)
        cv.imshow('cascade_500p_1000n_10s_high_hr_image', cascade_500p_1000n_10s_high_hr_image)
        cv.imshow('cascade_500p_1000n_10s_low_hr_image', cascade_500p_1000n_10s_low_hr_image)

        cv.waitKey(0)

else:
    print("Invalid option...terminating")
    exit(0)