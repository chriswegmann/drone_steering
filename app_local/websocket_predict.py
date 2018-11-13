# import required libraries
import tello
import threading
import h5py
import numpy as np
import pandas as pd
import time
import json
import websockets
import asyncio
from keras.models import load_model
from pipeline import XCentralizer, YCentralizer, YScaler
from sklearn.pipeline import make_pipeline, make_union
import warnings
warnings.filterwarnings("ignore")

# set debugging parameters
virtual_flight = True  # flight commands are printed, but not sent to drone
model_type = 'delta'  # allowed values are 'delta', 'posture', 'gesture'

# connect to drone and set status flags
drone_last_action = time.time()
drone_status = 'grounded'
if not virtual_flight:
    drone = tello.Tello('192.168.10.2', 8888)

movements = {0: 'takeoff',
             1: 'move_forward',
             2: 'flip',
             3: 'rotate_cw',
             4: 'rotate_ccw',
             5: 'land',
             999: 'not detected'}

# load model and initiate pipeline
if ((model_type == 'posture') | (model_type == 'gesture')):

    model = load_model('../models/model_' + model_type + '.h5')

    cols_x = ['leftShoulder_x',
              'rightShoulder_x',
              'leftWrist_x',
              'rightWrist_x',
              'leftHip_x',
              'rightHip_x']

    cols_y = [col.replace('x', 'y') for col in cols_x]

    processing_pipeline = make_pipeline(XCentralizer(cols_x),
                                        YCentralizer(cols_y),
                                        YScaler())


async def consumer_handler(websocket, path):

    global model_type
    print(model_type)

    print('Accepting incoming snapshots. Waiting for take-off signal.')
    async for pose_json in websocket:
        pose_dict = json_to_dict(pose_json)
        if model_type == 'delta':
            steer_drone(predict_movement_delta(pose_dict))
        if model_type == 'posture':
            steer_drone(predict_movement_model_posture(pose_dict))
        if model_type == 'gesture':
            steer_drone(predict_movement_model_gesture(pose_dict))


def steer_drone(movement):
    global drone_last_action
    global drone_status

    if ((movement == 0) & (drone_status == 'grounded')):
        threading.Thread(target=drone_takeoff).start()
        print('Take-off initiated. Ready to take flight commands in five seconds.')
        drone_last_action = time.time()
        time.sleep(4)
    if ((drone_status != 'grounded') & ((time.time() - drone_last_action) > 1.7)):
        drone_last_action = time.time()
        if movement == 1:
            threading.Thread(target=drone_move_forward).start()
        if movement == 2:
            threading.Thread(target=drone_flip).start()
        if movement == 3:
            threading.Thread(target=drone_rotate_cw).start()
        if movement == 4:
            threading.Thread(target=drone_rotate_ccw).start()
        if movement == 5:
            threading.Thread(target=drone_land).start()


def drone_takeoff():
    global drone_status
    drone_status = 'flying'
    print('drone.takeoff()')
    print("drone_status = 'flying'")
    if not virtual_flight:
        drone.takeoff()


def drone_move_forward():
    print('drone.move_forward(2)')
    if not virtual_flight:
        drone.move_forward(2)


def drone_flip():
    print("drone.flip('r')")
    if not virtual_flight:
        drone.flip('r')


def drone_rotate_cw():
    print('drone.rotate_cw(90)')
    if not virtual_flight:
        drone.rotate_cw(90)


def drone_rotate_ccw():
    print('drone.rotate_ccw(90)')
    if not virtual_flight:
        drone.rotate_ccw(90)


def drone_land():
    global drone_status
    drone_status = 'grounded'
    print('drone.land()')
    print("drone_status = 'grounded'")
    if not virtual_flight:
        drone.land()


def json_to_dict(pose_json):

    x = json.loads(pose_json)
    pose_dict = {}
    for i in range(8):
        pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                  '_x'] = x['poses'][0]['keypoints'][i + 5]['position']['x']
        pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                  '_y'] = x['poses'][0]['keypoints'][i + 5]['position']['y']

    del pose_dict['leftElbow_x']
    del pose_dict['leftElbow_y']
    del pose_dict['rightElbow_x']
    del pose_dict['rightElbow_y']

    return pose_dict


def predict_movement_delta(pose_dict):

    movement = 999  # no movement detected

    leftArm_x = pose_dict['leftWrist_x'] - pose_dict['leftShoulder_x']
    rightArm_x = pose_dict['rightShoulder_x'] - pose_dict['rightWrist_x']
    leftArm_y = pose_dict['leftShoulder_y'] - pose_dict['leftWrist_y']
    rightArm_y = pose_dict['rightShoulder_y'] - pose_dict['rightWrist_y']

    vertical_threshold = 50
    horizontal_threshold = 40
    undetected_threshold = 30

    if ((leftArm_y > vertical_threshold) & (rightArm_y > vertical_threshold) & (abs(leftArm_x) < undetected_threshold) & (abs(rightArm_x) < undetected_threshold)):
        movement = 0  # takeoff

    if ((abs(leftArm_y) < undetected_threshold) & (abs(rightArm_y) < undetected_threshold) & (leftArm_x > horizontal_threshold) & (rightArm_x > horizontal_threshold)):
        movement = 1  # move_forward

    if ((abs(leftArm_x) < undetected_threshold) & (abs(rightArm_x) < undetected_threshold) & (abs(leftArm_y) < undetected_threshold) & (abs(rightArm_y) < undetected_threshold)):
        movement = 2  # flip

    if ((leftArm_y < -vertical_threshold) & (abs(rightArm_y) < undetected_threshold) & (abs(leftArm_x) < undetected_threshold) & (rightArm_x > horizontal_threshold)):
        movement = 3  # rotate_cw

    if ((abs(leftArm_y) < undetected_threshold) & (rightArm_y < -vertical_threshold) & (leftArm_x > horizontal_threshold) & (abs(rightArm_x) < undetected_threshold)):
        movement = 4  # rotate_ccw

    if ((leftArm_y < -vertical_threshold) & (rightArm_y < -vertical_threshold) & (abs(leftArm_x) < undetected_threshold) & (abs(rightArm_x) < undetected_threshold)):
        movement = 5  # land

    return movement


def predict_movement_model_posture(pose_dict):

    pose_df = pd.DataFrame(pose_dict, index=[0])
    processing_pipeline.fit_transform(pose_df)
    movement = np.argmax(model.predict(pose_df)[0])

    return movement


def predict_movement_model_gesture(pose_dict):

    movement = 0

    return movement


# start websocket server
start_server = websockets.serve(consumer_handler, 'localhost', 8080)
print('Websocket server started. Please connect PoseNet on port 8080.')

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
