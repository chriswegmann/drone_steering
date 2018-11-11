import tello
import time

drone = tello.Tello('192.168.10.2', 8888)
drone.takeoff()
time.sleep(5)

start = time.time()
drone.flip('r')
end = time.time()
print("drone.flip('r')" + str(end - start))

time.sleep(5)
drone.land()
