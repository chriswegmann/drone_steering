import asyncio
import websockets
import json
import time
import pandas as pd
import numpy as np
import h5py

from keras.models import load_model
from pipeline import Shuffler, XCentralizer, YCentralizer, YScaler
from sklearn.pipeline import make_pipeline, make_union

movements = {0:'stop', 1:'left', 2:'right', 3:'up', 4:'down'}
model = load_model('models/pose_model.h5')

x_cols = ['leftShoulder_x','rightShoulder_x','leftElbow_x','rightElbow_x','leftWrist_x','rightWrist_x','leftHip_x','rightHip_x']
y_cols = ['leftShoulder_y','rightShoulder_y','leftElbow_y','rightElbow_y','leftWrist_y','rightWrist_y','leftHip_y','rightHip_y']
processing_pipeline = make_pipeline(XCentralizer(x_cols),YCentralizer(y_cols),YScaler(),Shuffler())

async def consumer_handler(websocket, path):
    print('Accepting incoming poses.')
    async for pose in websocket:
        print_pose(pose)
        #predict_label(pose)


def predict_label(pose):
# predict the probability of movements for a pose
    x = json.loads(pose)
    pose_df = {}
    for i in range(len(x['poses'][0]['keypoints'])):
        pose_df[x['poses'][0]['keypoints'][i]['part']+'_x'] = round(x['poses'][0]['keypoints'][i]['position']['x'],2) / 800
        pose_df[x['poses'][0]['keypoints'][i]['part']+'_y'] = round(x['poses'][0]['keypoints'][i]['position']['y'],2) / 800
        pose_df = pd.DataFrame(pose_df, index=[0])

    pose_df = pose_df.drop(columns=['nose_x','nose_y',
                        'leftEye_x','leftEye_y',
                        'rightEye_x','rightEye_y',
                        'leftEar_x','leftEar_y',
                        'rightEar_x','rightEar_y',
                        'leftKnee_x','leftKnee_y',
                        'rightKnee_x','rightKnee_y',
                        'leftAnkle_x','leftAnkle_y',
                        'rightAnkle_x','rightAnkle_y'])

    processing_pipeline.fit_transform(pose_df)
    #print(movements[np.argmax(model.predict(pose_df)[0])])
    
    movement = np.argmax(model.predict(pose_df)[0])
    print(movements[movement])
    
    file = open('pose_' + str(time.time()) + '.json','w')
    file.write(str(movement) + '\n')
    file.write(pose)
    file.close()



def print_pose(pose):
# print all x/y coordinates of a pose

    x = json.loads(pose)
    print(x['poses'][0]['score'])
    print('')
    for i in range(len(x['poses'][0]['keypoints'])):
        print(x['poses'][0]['keypoints'][i]['part'] + ': ' + \
            str(round(x['poses'][0]['keypoints'][i]['position']['x'],2)) + ' / ' + \
            str(round(x['poses'][0]['keypoints'][i]['position']['y'],2)))



start_server = websockets.serve(consumer_handler, 'localhost', 8081)
print('Websocket server started.')

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
