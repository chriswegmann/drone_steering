import pandas as pd
import time
import tello

df = pd.read_csv('dataframe.csv')

interval = 0.130
drone_last_action = time.time()
time.sleep(1.2)  # needed for first action to be taken into account
drone_status = 'grounded'
movements = ['looping', 'left', 'right', 'up', 'take-off', 'move']

drone = tello.Tello('192.168.10.2', 8888)


def steer_drone(movement):
    global drone_last_action
    global drone_status
    if (time.time() - drone_last_action) > 1:
        drone_last_action = time.time()
        if ((movement == 4) & (drone_status == 'grounded')):
            drone_status = 'flying'
            print('drone.takeoff()')
            drone.takeoff()
        if (drone_status != 'grounded'):
            if movement == 6:
                drone_status = 'grounded'
                print('drone.land()')
                drone.land()
            if movement == 0:
                print("drone.flip('r')")
                drone.flip('r')
            if movement == 1:
                print('drone.rotate_ccw(45)')
                drone.rotate_ccw(45)
            if movement == 2:
                print('drone.rotate_cw(45)')
                drone.rotate_cw(45)
            if movement == 5:
                print('drone.move_forward(1)')
                drone.move_forward(1)


def predict_movement(pose):

    movement = 5

    leftArm_x = pose['leftWrist_x'] - pose['leftShoulder_x']
    rightArm_x = pose['rightShoulder_x'] - pose['rightWrist_x']
    leftArm_y = pose['leftShoulder_y'] - pose['leftWrist_y']
    rightArm_y = pose['rightShoulder_y'] - pose['rightWrist_y']

    if leftArm_x > 60:
        movement = 1

    if rightArm_x > 60:
        movement = 2

    if ((leftArm_x > 60) & (rightArm_x > 60)):
        movement = 0

    if ((leftArm_y > 100) & (rightArm_y > 100)):
        movement = 3

    if ((leftArm_y < -100) & (rightArm_y < -100)):
        movement = 4

    return movement


for index, pose in df.iterrows():
    steer_drone(predict_movement(pose))
    time.sleep(interval)
print('drone.land()')
drone_status = 'grounded'
