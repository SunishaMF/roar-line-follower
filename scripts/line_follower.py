#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from carla_msgs.msg import CarlaEgoVehicleControl
from collections import deque
from sensor_msgs.msg import Image, CameraInfo


class Follower:
    def __init__(self):

        self.lateral_queue_length = 10
        self.lateral_error_queue = deque(maxlen=self.lateral_queue_length)
        self.K = {
            "Kp" : 0.85,
            "Ki" : 0,
            "Kd" : 0.1
        } # PID Parameters
        # For Car 03 P=0.85, D=0.1
        # For Car 02 P

        # K_mag = (self.K["Kp"] **2 + self.K["Ki"] **2 + self.K["Kd"] **2)**0.5
        # self.K["Kp"] /= K_mag
        # self.K["Ki"] /= K_mag
        # self.K["Kd"] /= K_mag

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('rgb_image', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd_manual', 
                                            CarlaEgoVehicleControl, queue_size=1)
        self.control = CarlaEgoVehicleControl()

    def calculate_moments(self, image, mask, y_offset=0.6, window_size=30):
        #calculating height and width
        h, w, d = image.shape
        search_top = y_offset*h
        search_bot = search_top + window_size
        curr_mask = mask.copy()
        curr_mask[0:int(search_top), 0:w] = 0
        curr_mask[int(search_bot):h, 0:w] = 0
        kernel = np.ones((5, 5), np.uint8)
        curr_mask = cv2.erode(curr_mask, kernel, iterations=1)
        curr_mask = cv2.dilate(curr_mask, kernel, iterations=1)
        M = cv2.moments(curr_mask)
        return M, curr_mask

    def calculate_P_err(self, centroid, width):
        cx, cy = centroid

        #Current Lateral Error of Centroid from Image Centre - for P
        # WRITE BETTER COMMENTS HERE
        lateral_error = float(cx - width/2 + width/20)
        return lateral_error

    def calculate_D_err(self,lateral_error):
        if len(self.lateral_error_queue) == 0:
            derivative_lateral_error = 0 # If queue is empty no derivative
        else:
            #If queue is not empty, then current - previous is derivative
            derivative_lateral_error = lateral_error - self.lateral_error_queue[-1]
        return derivative_lateral_error
    
    def err_at_furthest_pt(self, image, mask, y_offset_init=0.7, step_size=0.05, window_size=30, i = 0):
        print(i)
        h, w, d = image.shape

        M, curr_mask = self.calculate_moments(image, mask, y_offset=y_offset_init, window_size=window_size)

        if  M['m00'] > 0:
            #We have found line at this offset

            # Calculate the centroid 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # Put a circle on the image at centroid
            cv2.circle(image, (cx,cy), 20, (0,0,0), -1)

            P_error = self.calculate_P_err((cx,cy), w)
            I_error = sum(self.lateral_error_queue)
            D_error = self.calculate_D_err(P_error)

            # Calculating total PID Error
            normalized_total_error = self.K["Kp"]*P_error/(w/2) + \
                                    self.K["Ki"]*I_error/(self.lateral_queue_length*w/2) + \
                                    self.K["Kd"]*D_error/(w)

            clipped_total_error = np.clip(normalized_total_error, -1, 1)
            # #Appending Current Error to Queue for next iteration
            # self.lateral_error_queue.append(P_error)
            cv2.imshow("curr_mask", curr_mask)
            return(clipped_total_error)

        else:
            #We did not find line at this offset try another
            if y_offset_init + step_size <= 1:
                return self.err_at_furthest_pt(image, mask, y_offset_init=y_offset_init+step_size, step_size=0.05, window_size=30, i=i+1)
            else:
                return None

    def image_callback(self, msg):

        # Converting ros img message to RGB
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        # Converting RGB to HSV
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        '''Since cv2 has two ranges for red in the HSV Color Space,
            we define two different masks and combine them.'''
        # Creating mask for the lower end of HSV red
        lower_hue_minima = np.array([  0, 100,  200])
        lower_hue_maxima = np.array([ 10, 255, 255])
        lower_mask = cv2.inRange(converted_image, lower_hue_minima, lower_hue_maxima)

        # Creating mask for the lower end of HSV red
        upper_hue_minima = np.array([160, 100,  200])
        upper_hue_maxima = np.array([179, 255, 255])
        upper_mask = cv2.inRange(converted_image, upper_hue_minima, upper_hue_maxima)

        # # Creating mask for HSV yellow
        # yellow_minima = np.array([ 25, 100, 200])
        # yellow_maxima = np.array([ 35, 255, 255])
        # yellow_mask = cv2.inRange(converted_image, yellow_minima, yellow_maxima)

        # Combining the masks
        mask = cv2.bitwise_or(lower_mask, upper_mask)
        # mask = cv2.bitwise_or(mask, yellow_mask)

        full_mask = mask.copy()

        total_error = self.err_at_furthest_pt(image, mask, y_offset_init=0.71, step_size=0.05, window_size=30)

        if total_error is not None:
            # Line Found
            # Creating and Sending Control Message
            self.control.throttle = 0.0
            self.control.steer = total_error
            self.cmd_vel_pub.publish(self.control)
            #Appending Current Error to Queue for next iteration
            self.lateral_error_queue.append(total_error)
        # else:
        #     # Line not found

        cv2.imshow("full_mask",full_mask)
        # cv2.imshow("smooth_mask",mask_smooth)
        cv2.imshow("output", image)
        cv2.waitKey(3)

rospy.init_node('follower')
follower = Follower()
rospy.spin()