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
           'ms_since_last_frame',
           'ms_since_start']

df = pd.DataFrame(columns=columns)


async def consumer_handler(websocket, path):

    # store poses collected from previous run as a csv (as long as we don't know how to do this properly)
    global df
    df.to_csv('dataframe.csv',  index=False)
    df.drop(df.index, inplace=True)

    # capture incoming
    print('Accepting incoming snapshots')
    #stop = timeit.default_timer()
    async for pose_json in websocket:
        if 'start' in locals():
            start = stop
        else:
            start_fixed = timeit.default_timer()
            start = timeit.default_timer()
        features = json_to_dict(pose_json)
        stop = timeit.default_timer()
        ms_since_last_frame = str(round(1000*(stop - start)))
        ms_since_start = str(round(1000*(stop - start_fixed)))

        # features['label'] = predict_movement_delta(json_to_dict(pose_json))
        features['ms_since_last_frame'] = ms_since_last_frame
        features['ms_since_start'] = ms_since_start
        df = df.append(pd.DataFrame(features, index=[0]))

        print('Snapshot captured in ' + ms_since_last_frame + 'ms')
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

    dist_vertical = 50
    dist_horizontal = 40
    dist_no_mov = 30

    leftArm_x = pose_dict['leftWrist_x'] - pose_dict['leftShoulder_x']
    rightArm_x = pose_dict['rightShoulder_x'] - pose_dict['rightWrist_x']
    leftArm_y = pose_dict['leftShoulder_y'] - pose_dict['leftWrist_y']
    rightArm_y = pose_dict['rightShoulder_y'] - pose_dict['rightWrist_y']

    if ((leftArm_y > dist_vertical) & (rightArm_y > dist_vertical) & (abs(leftArm_x) < dist_no_mov) & (abs(rightArm_x) < dist_no_mov)):
        movement = 0  # takeoff

    if ((abs(leftArm_y) < dist_no_mov) & (abs(rightArm_y) < dist_no_mov) & (leftArm_x > dist_horizontal) & (rightArm_x > dist_horizontal)):
        movement = 1  # move_forward

    if ((abs(leftArm_x) < dist_no_mov) & (abs(rightArm_x) < dist_no_mov) & (abs(leftArm_y) < dist_no_mov) & (abs(rightArm_y) < dist_no_mov)):
        movement = 2  # flip

    if ((leftArm_y > dist_horizontal) & (abs(rightArm_y) < dist_no_mov) & (abs(leftArm_x) < dist_no_mov) & (rightArm_x > dist_horizontal)):
        movement = 3  # rotate_cw

    if ((abs(leftArm_y) < dist_no_mov) & (rightArm_y > dist_horizontal) & (leftArm_x > dist_horizontal) & (abs(rightArm_x) < dist_no_mov)):
        movement = 4  # rotate_ccw

    if ((leftArm_y < -dist_vertical) & (rightArm_y < -dist_vertical) & (abs(leftArm_x) < dist_no_mov) & (abs(rightArm_x) < dist_no_mov)):
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
