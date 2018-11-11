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
from pipeline import Shuffler, XCentralizer, YCentralizer, YScaler
from sklearn.pipeline import make_pipeline, make_union
import warnings
warnings.filterwarnings("ignore")


# connect to drone
drone = tello.Tello('192.168.10.2', 8888)
drone_last_action = time.time()
drone_status = 'grounded'

# time.sleep(1.2)  # needed for first action to be taken into account

movements = {0: 'takeoff',
             1: 'move_forward',
             2: 'flip',
             3: 'rotate_cw',
             4: 'rotate_ccw',
             5: 'land',
             999: 'not detected'}

model = load_model('../models/drone_pos_model-nonpipeline.h5')

x_cols = ['leftShoulder_x',
          'rightShoulder_x',
          'leftElbow_x',
          'rightElbow_x',
          'leftWrist_x',
          'rightWrist_x',
          'leftHip_x',
          'rightHip_x']

y_cols = ['leftShoulder_y',
          'rightShoulder_y',
          'leftElbow_y',
          'rightElbow_y',
          'leftWrist_y',
          'rightWrist_y',
          'leftHip_y',
          'rightHip_y']

processing_pipeline = make_pipeline(XCentralizer(
    x_cols), YCentralizer(y_cols), YScaler(), Shuffler())


async def consumer_handler(websocket, path):

    print('Accepting incoming snapshots. Waiting for take-off signal.')
    async for pose_json in websocket:
        pose_dict = json_to_dict(pose_json)
        steer_drone(predict_movement_delta(pose_dict))
        # steer_drone(predict_movement_model_posture(pose_dict))
        # steer_drone(predict_movement_model_gesture(pose_dict))


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
    drone.takeoff()


def drone_move_forward():
    print('drone.move_forward(2)')
    drone.move_forward(2)


def drone_flip():
    print("drone.flip('r')")
    drone.flip('r')


def drone_rotate_cw():
    print('drone.rotate_cw(90)')
    drone.rotate_cw(90)


def drone_rotate_ccw():
    print('drone.rotate_ccw(90)')
    drone.rotate_ccw(90)


def drone_land():
    global drone_status
    drone_status = 'grounded'
    print('drone.land()')
    print("drone_status = 'grounded'")
    drone.land()


def json_to_dict(pose_json):

    x = json.loads(pose_json)
    pose_dict = {}
    for i in range(8):
        pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                  '_x'] = x['poses'][0]['keypoints'][i + 5]['position']['x']
        pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                  '_y'] = x['poses'][0]['keypoints'][i + 5]['position']['y']

    return pose_dict


def predict_movement_delta(pose_dict):

    movement = 999

    leftArm_x = pose_dict['leftWrist_x'] - pose_dict['leftShoulder_x']
    rightArm_x = pose_dict['rightShoulder_x'] - pose_dict['rightWrist_x']
    leftArm_y = pose_dict['leftShoulder_y'] - pose_dict['leftWrist_y']
    rightArm_y = pose_dict['rightShoulder_y'] - pose_dict['rightWrist_y']

    if ((leftArm_y > 100) & (rightArm_y > 100) & (abs(leftArm_x) < 30) & (abs(rightArm_x) < 30)):
        movement = 0  # takeoff

    if ((abs(leftArm_y) < 30) & (abs(rightArm_y) < 30) & (leftArm_x > 60) & (rightArm_x > 60)):
        movement = 1  # move_forward

    if ((abs(leftArm_x) < 30) & (abs(rightArm_x) < 30) & (abs(leftArm_y) < 30) & (abs(rightArm_y) < 30)):
        movement = 2  # flip

    if ((leftArm_y < -100) & (abs(rightArm_y) < 30) & (abs(leftArm_x) < 30) & (rightArm_x > 60)):
        movement = 3  # rotate_cw

    if ((abs(leftArm_y) < 30) & (rightArm_y < -100) & (leftArm_x > 60) & (abs(rightArm_x) < 30)):
        movement = 4  # rotate_ccw

    if ((leftArm_y < -100) & (rightArm_y < -100) & (abs(leftArm_x) < 30) & (abs(rightArm_x) < 30)):
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
