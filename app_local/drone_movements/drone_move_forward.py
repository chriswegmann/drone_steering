import tello
import time

drone = tello.Tello('192.168.10.2', 8888)
drone.takeoff()
time.sleep(5)

start = time.time()
drone.move_forward(2)
end = time.time()
print('drone.move_forward(2)' + str(end - start))

time.sleep(5)
drone.land()
