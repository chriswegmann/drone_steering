from datetime import datetime
import numpy as np
import pandas as pd
import timeit
import json
import websockets
import asyncio
import warnings
warnings.filterwarnings("ignore")

columns_coord = ['leftShoulder_x',
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
                 'rightHip_y']

columns_time = ['ms_since_last_frame',
                'ms_since_start']

columns = columns_coord + columns_time

df = pd.DataFrame(columns=columns)


async def consumer_handler(websocket, path):
    # adds coordinates to a dataframe and saves it after the connection is closed

    global df
    df.to_csv('dataframe.csv',  index=False)
    df.drop(df.index, inplace=True)

    print('Accepting incoming snapshots')
    try:
        async for pose_json in websocket:
            if 'start' in locals():
                start = stop
            else:
                start = timeit.default_timer()
            features = json_to_dict(pose_json)
            stop = timeit.default_timer()
            if features:
                ms_since_last_frame = str(round(1000*(stop-start)))
                features['ms_since_last_frame'] = ms_since_last_frame
                df = df.append(pd.DataFrame(features, index=[0]))
                print('Snapshot captured in ' + ms_since_last_frame + 'ms')
            else:
                print('No wireframe data detected.')

    except:
        file_name = 'dataframe_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        df.drop_duplicates(subset=columns_coord, inplace=True)
        df.to_csv('../data/' + file_name,  index=False)

        df = pd.read_csv('../data/' + file_name)
        ms_since_start = 0
        for index, row in df.iterrows():
            ms_since_start += row['ms_since_last_frame']
            df.iloc[index, df.columns.get_loc(
                'ms_since_start')] = ms_since_start
        df.to_csv('../data/' + file_name,  index=False)

        print('Data saved in file ../data/' + file_name + '.')
        print('Websocket connection terminated. Please re-connect.')


def json_to_dict(pose_json):
    # converts the poses from json format to a subset in a dictionary

    x = json.loads(pose_json)
    if (len(x['poses']) == 0):
        return False
    else:
        pose_dict = {}
        for i in range(8):
            pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                      '_x'] = x['poses'][0]['keypoints'][i + 5]['position']['x']
            pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                      '_y'] = x['poses'][0]['keypoints'][i + 5]['position']['y']
        return pose_dict


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
