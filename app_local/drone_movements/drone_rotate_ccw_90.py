import tello
import time

drone = tello.Tello('192.168.10.2', 8888)
drone.takeoff()
time.sleep(5)

start = time.time()
drone.rotate_ccw(90)
end = time.time()
print('drone.rotate_ccw(90)' + str(end - start))

time.sleep(5)
drone.land()
