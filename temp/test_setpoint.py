# write script to connect to crazyflie drone and set setpoint

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
import time
URI = "radio://0/80/2M/E7E7E7E7E7"

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    print("Connected to Crazyflie")
    # scf.cf.commander.send_setpoint(0, 0, 0, 10000)
    time.sleep(2)