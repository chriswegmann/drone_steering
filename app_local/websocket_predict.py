from sklearn.pipeline import make_pipeline, make_union
from pipeline import Shuffler, XCentralizer, YCentralizer, YScaler
from keras.models import load_model
import asyncio
import websockets
import json
import time
import pandas as pd
import numpy as np
import h5py
import threading

import warnings
warnings.filterwarnings("ignore")

drone_last_action = time.time()
time.sleep(1.2)  # needed for first action to be taken into account
drone_status = 'grounded'

movements = {0: 'takeoff',
             1: 'move_forward',
             2: 'flip',
             3: 'rotate_cw',
             4: 'rotate_ccw',
             5: 'land',
             999: 'not detected'}

model = load_model('../models/drone_pos_model-nonpipeline.h5')

x_cols = ['leftShoulder_x', 'rightShoulder_x', 'leftElbow_x',
          'rightElbow_x', 'leftWrist_x', 'rightWrist_x', 'leftHip_x', 'rightHip_x']
y_cols = ['leftShoulder_y', 'rightShoulder_y', 'leftElbow_y',
          'rightElbow_y', 'leftWrist_y', 'rightWrist_y', 'leftHip_y', 'rightHip_y']
processing_pipeline = make_pipeline(XCentralizer(
    x_cols), YCentralizer(y_cols), YScaler(), Shuffler())


async def consumer_handler(websocket, path):

    print('Accepting incoming snapshots')
    async for pose_json in websocket:
        steer_drone(predict_movement_delta(json_to_dict(pose_json)))
        # steer_drone(predict_movement_model_posture(pose_json))


def steer_drone(movement):
    global drone_last_action
    global drone_status
    if (time.time() - drone_last_action) > 1.5:
        drone_last_action = time.time()
        if ((movement == 0) & (drone_status == 'grounded')):
            threading.Thread(target=drone_takeoff).start()
            print('time.sleep(5)')
            time.sleep(5)
        if (drone_status != 'grounded'):
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
    # drone.takeoff()


def drone_move_forward():
    print('drone.move_forward(2)')
    # time.sleep(1)
    # drone.move_forward(2)


def drone_flip():
    print("drone.flip('r')")
    # time.sleep(1)
    # drone.flip('r')


def drone_rotate_cw():
    print('drone.rotate_cw(45)')
    # time.sleep(1)
    # drone.rotate_cw(45)


def drone_rotate_ccw():
    print('drone.rotate_ccw(45)')
    # time.sleep(1)
    # drone.rotate_ccw(45)


def drone_land():
    global drone_status
    drone_status = 'grounded'
    print('drone.land()')
    print("drone_status = 'grounded'")
    # drone.land()


def predict_movement_delta(pose_dict):

    movement = 999

    leftArm_x = pose_dict['leftWrist_x'] - pose_dict['leftShoulder_x']
    rightArm_x = pose_dict['rightShoulder_x'] - pose_dict['rightWrist_x']
    leftArm_y = pose_dict['leftShoulder_y'] - pose_dict['leftWrist_y']
    rightArm_y = pose_dict['rightShoulder_y'] - pose_dict['rightWrist_y']

    # takeoff
    if ((leftArm_y > 100) & (rightArm_y > 100) & (abs(leftArm_x) < 30) & (abs(rightArm_x) < 30)):
        movement = 0

    # move_forward
    if ((abs(leftArm_y) < 30) & (abs(rightArm_y) < 30) & (leftArm_x > 60) & (rightArm_x > 60)):
        movement = 1

    # flip
    if ((abs(leftArm_x) < 30) & (abs(rightArm_x) < 30) & (abs(leftArm_y) < 30) & (abs(rightArm_y) < 30)):
        movement = 2

    # rotate_cw
    if ((leftArm_y < -100) & (abs(rightArm_y) < 30) & (abs(leftArm_x) < 30) & (rightArm_x > 60)):
        movement = 3

    # rotate_ccw
    if ((abs(leftArm_y) < 30) & (rightArm_y < -100) & (leftArm_x > 60) & (abs(rightArm_x) < 30)):
        movement = 4

    # land
    if ((leftArm_y < -100) & (rightArm_y < -100) & (abs(leftArm_x) < 30) & (abs(rightArm_x) < 30)):
        movement = 5

    return movement


def json_to_dict(pose_json):
    x = json.loads(pose_json)
    pose_dict = {}
    for i in range(8):
        pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                  '_x'] = x['poses'][0]['keypoints'][i + 5]['position']['x']
        pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                  '_y'] = x['poses'][0]['keypoints'][i + 5]['position']['y']
    return pose_dict


def predict_movement_model_posture(pose_json):
    # predict the probability of movements for a pose
    x = json.loads(pose_json)
    pose_df = {}
    for i in range(len(x['poses'][0]['keypoints'])):
        pose_df[x['poses'][0]['keypoints'][i]['part'] +
                '_x'] = round(x['poses'][0]['keypoints'][i]['position']['x'], 2) / 800
        pose_df[x['poses'][0]['keypoints'][i]['part'] +
                '_y'] = round(x['poses'][0]['keypoints'][i]['position']['y'], 2) / 800
        pose_df = pd.DataFrame(pose_df, index=[0])

    pose_df = pose_df.drop(columns=['nose_x', 'nose_y',
                                    'leftEye_x', 'leftEye_y',
                                    'rightEye_x', 'rightEye_y',
                                    'leftEar_x', 'leftEar_y',
                                    'rightEar_x', 'rightEar_y',
                                    'leftKnee_x', 'leftKnee_y',
                                    'rightKnee_x', 'rightKnee_y',
                                    'leftAnkle_x', 'leftAnkle_y',
                                    'rightAnkle_x', 'rightAnkle_y'])

    processing_pipeline.fit_transform(pose_df)
    # print(movements[np.argmax(model.predict(pose_df)[0])])

    movement = np.argmax(model.predict(pose_df)[0])
    print(movements[movement])

    file = open('pose_' + str(time.time()) + '.json', 'w')
    file.write(str(movement) + '\n')
    file.write(pose)
    file.close()


def print_pose(pose):
    # print all x/y coordinates of a pose

    x = json.loads(pose)
    print(x['poses'][0]['score'])
    print('')
    for i in range(len(x['poses'][0]['keypoints'])):
        print(x['poses'][0]['keypoints'][i]['part'] + ': ' +
              str(round(x['poses'][0]['keypoints'][i]['position']['x'], 2)) + ' / ' +
              str(round(x['poses'][0]['keypoints'][i]['position']['y'], 2)))


# start websocket server
start_server = websockets.serve(consumer_handler, 'localhost', 8080)
print('Websocket server started.')

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
