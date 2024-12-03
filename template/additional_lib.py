
from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import random
import cv2
import numpy as np

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import pandas as pd


from world import World
from keyboardControl import KeyboardControl
from hud import HUD
from sensors import *




try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name



# RGB Camera Callback
def rgb_callback(image, data_dict):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) #Reshaping with alpha channel
    img[:,:,3] = 255 #Setting the alpha to 255 
    data_dict['rgb_image'] = img


# Lidar Callback
def lidar_callback(image, data_dict):
    global res, lidar_range
    points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    lidar_data = np.array(points[:, :2])
    lidar_data *= min(res) / (2.0 * lidar_range)
    lidar_data += (0.5 * res[0], 0.5 * res[1])
    lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
    lidar_data = lidar_data.astype(np.int32)
    data_dict['lidar'] = lidar_data
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img_size = (res[0], res[1], 3)
    lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
    lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
    lidar_img = cv2.rotate(lidar_img, cv2.ROTATE_180)
    lidar_img = np.fliplr(lidar_img)
    data_dict['lidar_img'] = lidar_img


# Depth Camera Callback
def depth_callback(image, data_dict):
    # convert image to depth
    image.convert(cc.LogarithmicDepth)

    ############## RGB IMAGE ###############
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    ########################################

    # reshape array to 2D image array HxW
    array = np.reshape(array, (image.height, image.width, 4))
    # remove alpha channel
    array = array[:, :, :3]
    # convert to grayscale
    # array = array[:, :, ::-1]
    # convert to grayscale
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    data_dict['depth_image'] = array



# Calculate FPS
def calculate_fps():
    global frame_count, start_time
    frame_count += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        print("FPS: {:.2f}".format(fps))



########### PRINTING DATA ############
        '''
            The below code is used to print the data from the sensors mainly for testing purposes
        '''
        # print lidar data
        # print("Lidar: ", sensor_data['lidar'])

        # print control inputs
        # print(ego_vehicle.get_control())

        # IMU SENSOR
        # print("Accelerometer: ", imu.accelerometer)
        # print("Gyroscope: ", imu.gyroscope)
        # print("Compass: ", imu.compass)
        # print("\n\n")

        # GNSS SENSOR
        # print("Latitude: ", gnss.lat)
        # print("Longitude: ", gnss.lon)
        # print("\n\n")
        # print("Time: ", time)

        # Calculate FPS
        # calculate_fps()



        # Read a pandas file
        # data_file_path = foldername + 'dataIndex.csv'
        # df_existing = pd.read_csv(data_file_path)

        # print(df.accelerometer)
        # input("Press Enter to continue...")
        # print(df_existing.accelerometer[2])
        # print(type(df_existing.accelerometer[2]))
        # input("Press Enter to continue...")

        # Print the first few rows of the dataframe to verify
        # print(df_existing.head())



        # try:
        #     # show depth image
        #     cv2.imshow("Depth Image", sensor_data['depth_image'])    
        # except:

        #     pass



    # Initialize GNSS sensor
    # gnss = GnssSensor(ego_vehicle)

    # # Initialize Lidar sensor
    # lidar_bp = bp_lib.find('sensor.lidar.ray_cast')

    # # set the range of the lidar sensor
    # lidar_bp.set_attribute('range', str(lidar_range))

    # # lidar connecting to the world
    # lidar = world.spawn_actor(
    #                 lidar_bp,
    #                 carla.Transform(carla.Location(x=-0.1, z=1.7)),
    #                 attach_to=ego_vehicle,
    #                 attachment_type=Attachment.Rigid)

    # lidar.listen(lambda data: lidar_callback(data, sensor_data))


    # Get depth camera
    # bp_depth = bp_lib.find('sensor.camera.depth')
    # bp_depth.set_attribute('image_size_x', str(image_w))
    # bp_depth.set_attribute('image_size_y', str(image_h))

    # # depth camera connecting to the world
    # depth_camera = world.spawn_actor(
    #                 bp_depth,
    #                 carla.Transform(carla.Location(x=-0.1, z=1.7)),
    #                 attach_to=ego_vehicle,
    #                 attachment_type=Attachment.Rigid)

    # depth_trans = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)) 
    # depth_camera = world.spawn_actor(
    #                 bp_depth, 
    #                 depth_trans, 
    #                 attach_to=ego_vehicle, 
    #                 attachment_type=carla.AttachmentType.SpringArm)


    # depth_trans = carla.Transform(carla.Location(x=1.6, z=1.7)) 
    # depth_camera = world.spawn_actor(
    #                 bp_depth, 
    #                 depth_trans, 
    #                 attach_to=ego_vehicle, 
    #                 attachment_type=carla.AttachmentType.Rigid)
    # listen
    # depth_camera.listen(lambda data: depth_callback(data, sensor_data))


            # if(SAVE):
        #     rgb_file = foldername + "rgb/rgb_" + str(frame_count) + ".png"
        #     cv2.imwrite(f'{rgb_file}', sensor_data['rgb_image'])

