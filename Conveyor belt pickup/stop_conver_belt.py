from utils.dobot_lib.DoBotArm import DoBotArm as Dbt
from utils.dobot_lib import DobotDllType as dType
import time


# --- Hailo detection: return ONE (label, u, v) ---
import threading
import time
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

HOME_POSE = (234.6, 13.5, 116.3) # X,Y,Z, rHead (adjust as needed)
bot=Dbt(HOME_POSE[0],HOME_POSE[1],HOME_POSE[2])
bot.moveHome()
bot.set_conveyor_speed(0, 1)

