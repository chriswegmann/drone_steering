import tello
import time

drone = tello.Tello('192.168.10.2', 8888)

print('drone.takeoff()')
drone.takeoff()
print('time.sleep(5)')
time.sleep(5)

print('drone.move_forward(2)')
drone.move_forward(2)

print("drone.flip('r')")
drone.flip('r')
print('time.sleep(1)')
time.sleep(1)

print("drone.rotate_cw(90)")
drone.rotate_cw(90)

print("drone.move_forward(2)")
drone.move_forward(2)

print("drone.rotate_cw(170)")
drone.rotate_cw(170)

print("drone.move_forward(5)")
drone.move_forward(5)

print('drone.land()')
drone.land()
