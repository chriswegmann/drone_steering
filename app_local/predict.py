# import required libraries
from module import XCentralizer, YCentralizer, YScaler, GestureTransformer
from keras.models import load_model
from datetime import datetime
import asyncio
import websockets
import json
import time
import timeit
import pandas as pd
import numpy as np
import h5py
import threading
import tello
from sklearn.pipeline import make_pipeline, make_union
import math
import warnings
from scipy.interpolate import interp1d
from os import listdir
import re

warnings.filterwarnings("ignore")

# get model type
print('')
print("Which model type do you want to use? 1 = delta, 2 = posture, 3 = gesture")
model_type_id = input()
if int(model_type_id) == 1:
    model_type = 'delta'
if int(model_type_id) == 2:
    model_type = 'posture'
if int(model_type_id) == 3:
    model_type = 'gesture'
print('The "' + model_type + '" model will be used.')

# get interpolation yes / no
if int(model_type_id) == 3:
    print('')
    print("Do you want to use interpolation? y = yes, n = no")
    use_interpolation_id = input()
    if str(use_interpolation_id)=='y':
        use_interpolation = True
        print('The PoseNet wireframes will be interpolated.')
        interpolation = 'ip'
    else:
        use_interpolation = False
        print('The PoseNet wireframes will not be interpolated.')
        interpolation = 'nip'
print('')

# get model instance
if int(model_type_id) == 3:
    file_names = listdir('../models/')
    pattern = '(?P<model_type>model_' + model_type + '{1}_.*)(?P<ip_nip>' + interpolation + '{1})(?P<model_parameters>.+)(?P<suffix>.h5)'
    reg = re.compile(pattern)
    matches = []
    for file_name in file_names:
        match = reg.search(file_name)
        if match:
            matches.append(match)
    models = []
    for i, match in enumerate(matches):
        model = match.groupdict()
        models.append(model['model_type'] + model['ip_nip'] + model['model_parameters'])
    print('Which model instance do you want to use?')
    for i in range(len(models)):
        print(str(i) + ' | ' + models[i])
    model_instance = input()
    print('')

# decide if drone or virtual flight is used
print("Do you connect to the drone or do a virtual flight? d = drone, v = virtual flight")
virtual_flight_id = input()
if str(virtual_flight_id)=='d':
    virtual_flight = False
    print('Connecting to drone...')
else:
    virtual_flight = True
    print('You will see the flight commands printed on the screen.')
print('')

# set interpolation parameters
ms_per_frame_original = 120
gesture_length = 2000
ms_per_frame_interpolated = 50
add_interpol_frames = 3

# connect to drone and set status flags
drone_last_action = time.time()
drone_status = 'grounded'
if not virtual_flight:
    drone = tello.Tello('192.168.10.2', 8888)

movements = {0: 'not detected',
             1: 'takeoff',
             2: 'move',
             3: 'flip',
             4: 'left',
             5: 'right',
             6: 'land'}

# load model and initiate pipeline
if (model_type == 'posture'):
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

if (model_type == 'gesture'):
    cols_x = ['leftShoulder_x',
              'rightShoulder_x',
              'leftWrist_x',
              'rightWrist_x',
              'leftHip_x',
              'rightHip_x',
              'leftElbow_x',
              'rightElbow_x']
    cols_y = [col.replace('x', 'y') for col in cols_x]
    model = load_model('../models/' + models[int(model_instance)] + '.h5')
    if use_interpolation:
        # model = load_model('../models/model_' + model_type + '_interpolation_' + str(ms_per_frame_interpolated) + '.h5')
        start_time = timeit.default_timer()
        ms_since_start = timeit.default_timer()
        cols = sorted(cols_x + cols_y + ['ms_since_start'])
        cols_without_ms = sorted(cols_x + cols_y)
        processing_pipeline = make_pipeline(GestureTransformer(byrow=True,feature_names=cols_without_ms))
    else:
        # model = load_model('../models/model_' + model_type + '_relaxed_labelling.h5')
        cols = sorted(cols_x + cols_y)
        processing_pipeline = make_pipeline(GestureTransformer(byrow=True,feature_names=cols))
    pose_df = pd.DataFrame(columns=cols)


async def consumer_handler(websocket, path):

    global model_type
    print(model_type)

    print('Accepting incoming snapshots. Waiting for take-off signal.')
    try:
        async for pose_json in websocket:
            if 'start' in locals():
                start = stop
            else:
                start = timeit.default_timer()
            stop = timeit.default_timer()
            ms_since_last_frame = str(round(1000*(stop-start)))
            print('Snapshot captured in ' + ms_since_last_frame + 'ms')

            pose_dict = json_to_dict(pose_json)
            if len(pose_dict) == 0:
                print('No wireframes detected. Please ensure that PoseNet detects wireframes.')
            else:
                if model_type == 'delta':
                    steer_drone(predict_movement_delta(pose_dict))
                if model_type == 'posture':
                    steer_drone(predict_movement_model_posture(pose_dict))
                if model_type == 'gesture':
                    steer_drone(predict_movement_model_gesture(pose_dict))
    except:
        print('Websocket connection terminated. Please re-connect.')
        #raise


def steer_drone(movement):
    global drone_last_action
    global drone_status

    if ((movement == 1) & (drone_status == 'grounded')):
        threading.Thread(target=drone_takeoff).start()
        drone_last_action = time.time()
        time.sleep(4)
    if ((drone_status != 'grounded') & ((time.time() - drone_last_action) > 1.7)):
        drone_last_action = time.time()
        if movement == 2:
            threading.Thread(target=drone_move).start()
        if movement == 3:
            threading.Thread(target=drone_flip).start()
        if movement == 4:
            threading.Thread(target=drone_left).start()
        if movement == 5:
            threading.Thread(target=drone_right).start()
        if movement == 6:
            threading.Thread(target=drone_land).start()


def drone_takeoff():
    global drone_status
    drone_status = 'flying'
    print('Take-off | drone.takeoff() | Ready to take flight commands in five seconds.')
    if not virtual_flight:
        drone.takeoff()


def drone_move():
    print('Move     | drone.move_forward(2)')
    if not virtual_flight:
        drone.move_forward(2)


def drone_flip():
    print("Flip     | drone.flip('r')")
    if not virtual_flight:
        drone.flip('r')


def drone_left():
    print('Left     | drone.rotate_ccw(90)')
    if not virtual_flight:
        drone.rotate_ccw(90)


def drone_right():
    print('Right    | drone.rotate_cw(90)')
    if not virtual_flight:
        drone.rotate_cw(90)


def drone_land():
    global drone_status
    drone_status = 'grounded'
    print('Land     | drone.land()')
    if not virtual_flight:
        drone.land()


def json_to_dict(pose_json):

    x = json.loads(pose_json)
    pose_dict = {}

    if len(x['poses']) > 0:
        for i in range(8):
            pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                    '_x'] = x['poses'][0]['keypoints'][i + 5]['position']['x']
            pose_dict[x['poses'][0]['keypoints'][i+5]['part'] +
                    '_y'] = x['poses'][0]['keypoints'][i + 5]['position']['y']

        if model_type != 'gesture':
            del pose_dict['leftElbow_x']
            del pose_dict['leftElbow_y']
            del pose_dict['rightElbow_x']
            del pose_dict['rightElbow_y']

    return pose_dict


def predict_movement_delta(pose_dict):

    movement = 0  # no movement detected

    leftArm_x = pose_dict['leftWrist_x'] - pose_dict['leftShoulder_x']
    rightArm_x = pose_dict['rightShoulder_x'] - pose_dict['rightWrist_x']
    leftArm_y = pose_dict['leftShoulder_y'] - pose_dict['leftWrist_y']
    rightArm_y = pose_dict['rightShoulder_y'] - pose_dict['rightWrist_y']

    vertical_threshold = 50
    horizontal_threshold = 40
    undetected_threshold = 30

    if ((leftArm_y > vertical_threshold) & (rightArm_y > vertical_threshold) & (abs(leftArm_x) < undetected_threshold) & (abs(rightArm_x) < undetected_threshold)):
        movement = 1  # takeoff

    if ((abs(leftArm_y) < undetected_threshold) & (abs(rightArm_y) < undetected_threshold) & (leftArm_x > horizontal_threshold) & (rightArm_x > horizontal_threshold)):
        movement = 2  # move

    if ((abs(leftArm_x) < undetected_threshold) & (abs(rightArm_x) < undetected_threshold) & (abs(leftArm_y) < undetected_threshold) & (abs(rightArm_y) < undetected_threshold)):
        movement = 3  # flip

    if ((leftArm_y < -vertical_threshold) & (abs(rightArm_y) < undetected_threshold) & (abs(leftArm_x) < undetected_threshold) & (rightArm_x > horizontal_threshold)):
        movement = 4  # left

    if ((abs(leftArm_y) < undetected_threshold) & (rightArm_y < -vertical_threshold) & (leftArm_x > horizontal_threshold) & (abs(rightArm_x) < undetected_threshold)):
        movement = 5  # right

    if ((leftArm_y < -vertical_threshold) & (rightArm_y < -vertical_threshold) & (abs(leftArm_x) < undetected_threshold) & (abs(rightArm_x) < undetected_threshold)):
        movement = 6  # land

    return movement


def predict_movement_model_posture(pose_dict):

    pose_df = pd.DataFrame(pose_dict, index=[0])
    processing_pipeline.fit_transform(pose_df)
    movement = np.argmax(model.predict(pose_df)[0])

    return movement


def predict_movement_model_gesture(pose_dict):

    global pose_df
    global start_time
    global ms_since_start
    movement = 0

    steps = math.ceil(gesture_length/ms_per_frame_original) + 1
    steps_ip = math.ceil(gesture_length/ms_per_frame_interpolated) + 1

    if use_interpolation:
        pose_new = pd.DataFrame(pose_dict, index=[0])
        pose_new.at[0,'ms_since_start'] = round(1000*(timeit.default_timer()-start_time))
        pose_df = pose_df.append(pose_new)
        pose_df.reset_index(inplace=True, drop=True)

        # keep only latest two seconds and the additional frames needed for interpolation
        cut_off = pose_df.iloc[pose_df['ms_since_start'].argmax()]['ms_since_start'] - 2000
        keep_df = pose_df.loc[pose_df['ms_since_start']>=cut_off]
        discarded_df = pose_df.loc[pose_df['ms_since_start']<cut_off]
        pose_df = pd.concat([discarded_df.tail(add_interpol_frames), keep_df]).reset_index(drop=True)

        if discarded_df.shape[0] > add_interpol_frames:
            pose_ip_df = interpolate(pose_df, ms_per_frame_interpolated)

            file_name = 'model_input_' + datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.csv'
            pose_ip_df.to_csv('model_inputs/' + file_name,  index=False)

            pose_np = pose_ip_df.drop(['ms_since_start'], axis=1).values.reshape(1, steps_ip, -1)
            pose_np = processing_pipeline.fit_transform(pose_np)
            movement = np.argmax(model.predict(pose_np)[0])
    else:
        pose_df = pose_df.append(pd.DataFrame(pose_dict, index=[0]))
        if len(pose_df) > steps:
            pose_df = pose_df.iloc[1:]

        if len(pose_df) == steps:
            file_name = 'model_input_' + datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.csv'
            pose_df.to_csv('model_inputs/' + file_name,  index=False)

            pose_np = pose_df.values.reshape(1, steps, len(cols))
            pose_np = processing_pipeline.fit_transform(pose_np)
            movement = np.argmax(model.predict(pose_np)[0])

    return movement


def interpolate(pose_df, ms_per_frame_interpolated):

    features = list(pose_df.filter(regex = '_(x|y)$', axis = 1).columns)
    t = pose_df["ms_since_start"].values

    interpolators = {}
    for feat in features:
        f = pose_df[feat].values
        
        cub_f = interp1d(t, f, 
                         kind = 'cubic', 
                         fill_value = (f[0],f[-1]), 
                         bounds_error = False,
                         assume_sorted=True)
        
        interpolators[feat] = cub_f
        
    n = int(np.ceil(gesture_length/ms_per_frame_interpolated) + 1)
    t_delta = sorted([ms_per_frame_interpolated * i for i in range(0,-n,-1)])
    t_new = t_delta + t[-1]

    interp_df = pd.DataFrame(columns = features)
    interp_df["ms_since_start"] = t_new

    for feat in features:
        interp_df[feat] = interpolators[feat](t_new)
    
    return interp_df


# start websocket server
start_server = websockets.serve(consumer_handler, 'localhost', 8080)
print('Websocket server started. Please connect PoseNet on port 8080.')

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
