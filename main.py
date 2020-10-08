import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from model import Net2
from hyperparameters import Hyp

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor

def main():
    image = cv2.imread('../images/obamas.jpg')

    # switch red and blue color channels 
    # --> by default OpenCV assumes BLUE comes first, not RED as in many images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("image shape = ", image.shape)
    # plot the image
    fig = plt.figure(figsize=(9,9))
    plt.imshow(image)
    plt.show()
    # load in a haar cascade classifier for detecting frontal faces
    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = get_faces(image)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

    fig = plt.figure(figsize=(9,9))

    plt.imshow(image_with_detections)
    #plt.show()
    # -------- Load model
    net = Net2()
    model_dir = '../saved_models/'
    model_name = 'keypoints_model_2.pt'
    data_transform = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        Normalize(),
        ToTensor()
    ])

    # retreive the saved model
    net_state_dict = torch.load(model_dir+model_name)
    net.load_state_dict(net_state_dict)
    print(net.eval())
    image_copy = np.copy(image)

    # loop over the detected faces from your haar cascade
    for (x,y,w,h) in faces:
        
        # Select the region of interest that is the face in the image 
        roi = image_copy[y:y+h, x:x+w]
        
        ## TODO: Convert the face region from RGB to grayscale
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        roi =  roi/255.0
        ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        roi = cv2.resize(roi, (224, 224))
        ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        image_reshape = torch.from_numpy(image.transpose((2, 0, 1)))
        ## TODO: Make facial keypoint predictions using your loaded, trained network 
        ## perform a forward pass to get the predicted facial keypoints
        keypoints = net.forward(image_reshape)
        ## TODO: Display each detected face and the corresponding keypoints                             
        show_keypoints(image_reshape, keypoints)

# helper function to display keypoints
def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    plt.show()

def net_sample_output(test_loader, net):
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts

def get_faces(image):
    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('../detector_architectures/haarcascade_frontalface_default.xml')

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    return face_cascade.detectMultiScale(image, 1.2, 2)

if __name__ == '__main__':    
    main()