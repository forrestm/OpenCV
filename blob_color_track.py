#! /usr/bin/env python

# Adapted from http://www.davidhampgonsalves.com/opencv-python-color-tracking/

##############################################################################
# WebcamColorTrack_MCHE470_Fall2013.py
#
# Script to track red in a webcam image
#
# Requires OpenCV
#
# Created: 11/2/13
#   - Joshua Vaughan
#   - joshua.vaughan@louisiana.edu
#   - http://www.ucs.louisiana.edu/~jev9637
#
# Modified:
#   * Forrest Montgomery
#   -edited the check variable on line 103 to work on Ubuntu
#
##############################################################################
# import cv2.cv as cv
import cv2
from numpy import *
from time import localtime, strftime
import time
import datetime

import matplotlib.pyplot as plt

color_tracker_window = "Tracking Window"

save_data = True

if save_data:
    # names the output file as the date and time that the program is run
    filename = strftime("%m_%d_%Y_%H%M")
    # gives the path of the file to be opened
    filepath = filename + ".txt"


class ColorTracker:

    def __init__(self):
        cv2.namedWindow(color_tracker_window, 1)
        # cv.NamedWindow(color_tracker_window, 1)
        self.capture = cv2.VideoCapture()
        # self.capture = cv.CaptureFromCAM(0)

    def cv_size(img):
        """Function to get image size"""
        return tuple(img.shape[1::-1])

    def create_blank(width, height):
        """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    def run(self):
        # sets the initial time
        initialTime = time.time()
        data = zeros((1, 3))

        while True:
            # img = cv.QueryFrame(self.capture)
            img = cv2.VideoCapture.read(self.capture)

            # blur the source image to reduce color noise
            # cv.Smooth(img, img, cv.CV_BLUR, 3)
            img = cv2.GaussianBlur(img, (15, 15), 0)

            # convert the image to hsv(Hue, Saturation, Value) so its
            # easier to determine the color to track(hue)
            img = cv2.imread('img')
            height, width, depth = img.shape
            hsv_img = np.zeros((height, width, 3), np.uint8)
            # hsv_img = cv.CreateImage(cv.GetSize(img), 8, 3)
            cv2.cvtColor(img, CV_BGR2HSV, hsv_img)
            # cv.CvtColor(img, hsv_img, cv.CV_BGR2HSV)

            # limit all pixels that don't match our criteria, in the is case we
            # are looking for purple but if you want you can adjust the first
            # value in both turples which is the hue range(120,140). OpenCV
            # uses 0-180 as a hue range for the HSV color model
            hsv_img = cv2.imread('img')
            thresholded_img = np.zeros((hsv_img.shape), np.uint8)
            # thresholded_img = cv.CreateImage(cv.GetSize(hsv_img), 8, 1)
            # cv.InRangeS(hsv_img, (112, 50, 50), (118, 200, 200),
            # thresholded_img)

            # try red
            # cv.InRangeS(hsv_img, (160, 150, 100), (180, 255, 255),
            #             thresholded_img)

            # try orange
            # cv.InRangeS(hsv_img, (14, 150, 100), (45, 255, 255),
                        # thresholded_img)
            orange_min = np.array([5, 50, 50], np.uint8)
            orange_max = np.array([15, 255, 255], np.uint8)
            thresholded_img = cv2.inRange(hsv_img, orange_min, orange_max)
            orange_binary = cv2.inRange(img, orange_min, orange_max)

            # Returns an array of given dimensions filled with 1s
            dilation = np.ones((15, 15), "uint8")
            # This causes the object to be increase in size
            red_binary = cv2.dilate(red_binary, dilation)
            # This finds the contours of a binary image
            contours, hierarchy = cv2.findContours(red_binary, cv2.RETR_LIST,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.contourArea(contour)
            # determine the objects moments and check that the area is large
            # enough to be our object
            # thresholded_img2 = cv.GetMat(thresholded_img)
            moments = cv2.moments(contour)
            # moments = cv.Moments(thresholded_img2, 0) ###### check####
            area = cv.GetCentralMoment(moments, 0, 0)

            # there can be noise in the video so ignore objects with small
            # areas
            if (area > 50000):
                # determine the x and y coordinates of the center of the object
                # we are tracking by dividing the 1, 0 and 0, 1 moments by
                # the area
                x = cv.GetSpatialMoment(moments, 1, 0) / area
                y = cv.GetSpatialMoment(moments, 0, 1) / area

                if save_data:
                    # Save the current time and pixel location into the data
                    # array
                    add = asarray([[time.time() - initialTime, x, y]])
                    data = append(data, add, 0)

                # convert center location to integers
                x = int(x)
                y = int(y)

                # create an overlay to mark the center of the tracked object
                overlay = cv.CreateImage(cv.GetSize(img), 8, 3)

                cv.Circle(overlay, (x, y), 2, (0, 0, 0), 20)
                cv.Add(img, overlay, img)

                # add the thresholded image back to the img so we can see what
                # was left after it was applied
                cv.Merge(thresholded_img2, None, None, None, img)

            # display the image
            cv.ShowImage(color_tracker_window, img)

            check = cv.WaitKey(1)

            # if check == 27 or check == 'ESC':
            # This number was figured out by pressing ESC on the keyboard
            # instead of 27 that was displayed
            if check == 1048603 or check == 'ESC':
                if save_data:
                    # save the data file as comma separated values
                    # remove the first row
                    data = delete(data, 0, 0)
                    savetxt(filepath, data, delimiter=",",
                            header=
                            "Time (s), X Position(pixels),Y Position(pixels)"
                            )
                    x = data[:, 1]
                    y = data[:, 2]
                    print data

                    plt.scatter(x, y)
                    plt.show()

                break


if __name__ == "__main__":
    color_tracker = ColorTracker()
    color_tracker.run()
