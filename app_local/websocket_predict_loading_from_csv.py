import pandas as pd
import time
import tello
import threading

df = pd.read_csv('dataframe_video_delta.csv')

interval = 0.130
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

drone = tello.Tello('192.168.10.2', 8888)


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
    drone.takeoff()


def drone_move_forward():
    print('drone.move_forward(2)')
    # time.sleep(1)
    drone.move_forward(2)


def drone_flip():
    print("drone.flip('r')")
    # time.sleep(1)
    drone.flip('r')


def drone_rotate_cw():
    print('drone.rotate_cw(45)')
    # time.sleep(1)
    drone.rotate_cw(45)


def drone_rotate_ccw():
    print('drone.rotate_ccw(45)')
    # time.sleep(1)
    drone.rotate_ccw(45)


def drone_land():
    global drone_status
    drone_status = 'grounded'
    print('drone.land()')
    print("drone_status = 'grounded'")
    drone.land()


def predict_movement(pose):

    movement = 999

    leftArm_x = pose['leftWrist_x'] - pose['leftShoulder_x']
    rightArm_x = pose['rightShoulder_x'] - pose['rightWrist_x']
    leftArm_y = pose['leftShoulder_y'] - pose['leftWrist_y']
    rightArm_y = pose['rightShoulder_y'] - pose['rightWrist_y']

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


for index, pose in df.iterrows():
    # print(movements[predict_movement(pose)])
    steer_drone(predict_movement(pose))
    time.sleep(interval)
