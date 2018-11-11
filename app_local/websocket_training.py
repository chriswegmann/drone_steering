import numpy as np
import pandas as pd
import timeit
import json
import websockets
import asyncio
import warnings
warnings.filterwarnings("ignore")


movements = {0: 'takeoff',
             1: 'move_forward',
             2: 'flip',
             3: 'rotate_cw',
             4: 'rotate_ccw',
             5: 'land',
             999: 'not detected'}

columns = ['leftShoulder_x',
           'leftShoulder_y',
           'rightShoulder_x',
           'rightShoulder_y',
           'leftElbow_x',
           'leftElbow_y',
           'rightElbow_x',
           'rightElbow_y',
           'leftWrist_x',
           'leftWrist_y',
           'rightWrist_x',
           'rightWrist_y',
           'leftHip_x',
           'leftHip_y',
           'rightHip_x',
           'rightHip_y',
           'label']

df = pd.DataFrame(columns=columns)


async def consumer_handler(websocket, path):

    # store poses collected from previous run as a csv (as long as we don't know how to do this properly)
    global df
    df.to_csv('dataframe.csv',  index=False)
    df.drop(df.index, inplace=True)

    # capture incoming
    print('Accepting incoming snapshots')
    async for pose_json in websocket:
        start = timeit.default_timer()
        features = json_to_dict(pose_json)
        features['label'] = predict_movement_delta(json_to_dict(pose_json))
        df = df.append(pd.DataFrame(features, index=[0]))
        stop = timeit.default_timer()
        print('Snapshot captured in ' + str(1000*(stop - start)) + 'ms')
        # print(pose_json)
        # print_pose(pose_json)


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


def print_pose(pose_json):
    # print all x/y coordinates of a pose in an easy-to-read format

    x = json.loads(pose_json)
    print(x['poses'][0]['score'])
    print('')
    for i in range(len(x['poses'][0]['keypoints'])):
        print(x['poses'][0]['keypoints'][i]['part'] + ': ' +
              str(round(x['poses'][0]['keypoints'][i]['position']['x'], 2)) + ' / ' +
              str(round(x['poses'][0]['keypoints'][i]['position']['y'], 2)))


# start websocket server
start_server = websockets.serve(consumer_handler, 'localhost', 8080)
print('Websocket server started. Please connect PoseNet on port 8080.')

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
