"""
Written by Forrest Montgomery, with help from Dr. Vaughan.

This program tracks two different colored objects, then spits out a graph and
a csv file of the postions of the objects. It will soon email the data to the
user of the accompanying website.
"""

import cv2
import numpy as np
from time import strftime
import time
import matplotlib.pyplot as plt

save_data = True

if save_data:
    # names the output file as the date and time that the program is run
    filename = strftime("%m_%d_%Y_%H%M")
    # gives the path of the file to be opened
    filepath = filename + ".txt"


class ColorTracker:

    def __init__(self):
        # Creates a window that can be used as a placeholder for images and
        # trackbars
        cv2.namedWindow("ColourTrackerWindow", cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow("ColourTrackerWindow2", cv2.CV_WINDOW_AUTOSIZE)
        # A class that captures video. The number is the id of the opened
        # video capturing device 0 is for the default device
        self.capture = cv2.VideoCapture(0)

    def run(self):
        # sets the inital time
        inital_Time = time.time()
        # Alots a zero numpy array to data, for storing the motion later
        data = np.zeros((1, 3))
        data2 = np.zeros((1, 3))

        while True:
            f, img = self.capture.read()
            # Just the raw footage from the camera
            original_image = img
            # Displays the raw footage
            cv2.imshow("ColourTrackerWindow", original_image)
            # Applies a Gassian Blur to the image
            img = cv2.GaussianBlur(img, (5, 5), 0)
            # Converts the color of the image to HSV colorspace
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # displays the img in the window defined by namedWindow
            cv2.imshow("ColourTrackerWindow", img)
            # Lower HSV color limit

            # try blue
            color2_lower = np.array([64, 113, 0], np.uint8)

            # try blue
            color2_upper = np.array([179, 255, 199], np.uint8)

            # try orange
            color_lower = np.array([24, 81, 214], np.uint8)
            # Upper HSV color limit

            # try orange
            color_upper = np.array([46, 255, 255], np.uint8)

            # Checks if array elements lie between the elements of two other
            # arrays

            thresholded_img_2 = cv2.inRange(img, color2_lower, color2_upper)

            thresholded_img = cv2.inRange(img, color_lower, color_upper)

            # This displays the theresholded image
            cv2.imshow("ColourTrackerWindow", thresholded_img)
            cv2.imshow("ColourTrackerWindow2", thresholded_img_2)
            moments = cv2.moments(thresholded_img)
            moments2 = cv2.moments(thresholded_img_2)
            area = moments['m00']
            area2 = moments2['m00']

            # there can be noise in the video so ignore objects with small
            # areas
            if (area > 100000) and (area2 > 100000):
                # determine the x and y coordinates of the center of the object
                # we are tracking by dividing the 1, 0 and 0, 1 moments by the
                # area
                x = moments['m10'] / area
                y = moments['m01'] / area
                x2 = moments2['m10'] / area2
                y2 = moments2['m01'] / area2

                if save_data:
                    # Save the current time and pixel location into the data
                    # array
                    add = np.asarray([[time.time() - inital_Time, x, y]])
                    add2 = np.asarray([[time.time() - inital_Time, x2, y2]])
                    data = np.append(data, add, 0)
                    data2 = np.append(data2, add2, 0)

                # convert center location to integers
                x = int(x)
                y = int(y)
                x2 = int(x2)
                y = int(y2)

                # Create an overlay to mark the center of the tracked object
                # img_shape = cv2.imread('img')
                # height, width, depth = img.shape
                # overlay = np.zeros((height, width, 3), np.uint8)

                # cv2.circle(overlay, (x, y), 2, (0, 0, 0), 20)
                # cv2.add(img, overlay, img)

                # img = cv2.merge(thresholded_img)

            # cv2.imshow("ColourTrackerWindow", img)
            check = cv2.waitKey(20)
            # This closes all the windows and shuts down the camera
            if check == 1048603 or check == 'ESC':
                if save_data:
                    # save the data file as comma separated values
                    # remove the first row
                    data = np.delete(data, 0, 0)
                    data2 = np.delete(data2, 0, 0)
                    # np.savetxt(filepath, data, delimiter=",", header="Time \
                    # (s), X Position(pixels), Y Position(pixels)")
                    x = data[:, 1]
                    # I flipped the camera so the negative is needed to make
                    # the graph look as if you are looking down.
                    y = np.negative(data[:, 2])
                    x2 = data2[:, 1]
                    y2 = np.negative(data2[:, 2])
                    print data
                    print data2

                    plt.figure(1)
                    plt.subplot(211)
                    plt.axis([0, 650, -250, 0])
                    plt.title('Orange')
                    plt.scatter(x, y)
                    plt.subplot(212)
                    plt.axis([0, 650, -250, 0])
                    plt.title('Blue')
                    plt.scatter(x2, y2)
                    plt.show()

                cv2.destroyWindow("ColourTrackerWindow")
                self.capture.release()
                break

if __name__ == "__main__":
    color_tracker = ColorTracker()
    color_tracker.run()
