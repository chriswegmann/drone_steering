import tello
import time

drone = tello.Tello('192.168.10.2', 8888)

start = time.time()
drone.takeoff()
end = time.time()
print('drone.takeoff()' + str(end - start))
