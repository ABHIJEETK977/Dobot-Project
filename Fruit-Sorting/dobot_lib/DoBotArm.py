#!/usr/bin/env python
## ** File Name : DoBotArm.py
import os
import sys

# Ensure DLL folder is in the path, using an absolute path
DLL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DLL'))
if DLL_DIR not in sys.path:
    sys.path.insert(0, DLL_DIR)

try:
    from . import DobotDllType as dType
except ImportError as e:
    print(f"ERROR: Could not import DobotDllType from {DLL_DIR}.")
    print("Make sure the DLL folder exists and contains DobotDllType.py and the required DLL files.")
    raise e

import time

"""-------The DoBot Control Class-------
Variables:
suction = Suction is currently on/off
picking: shows if the dobot is currently picking or dropping an item
api = variable for accessing the dobot .dll functions
home% = home position for %
                                  """

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

#Main control class for the DoBot Magician.
class DoBotArm:
    def __init__(self, homeX, homeY, homeZ,auto_disconnect=False):
        self.suction = False
        self.picking = False
        self.api = dType.load()
        self.homeX = homeX
        self.homeY = homeY
        self.homeZ = homeZ
        self.connected = False
        self.connected=self.dobotConnect()
        if not self.connected:
            raise RuntimeError("Failed to connect to DoBot (USB/cable/power/driver?)")

    def __del__(self):
        try:
            self.dobotDisconnect()
        except Exception:
            pass

    #Attempts to connect to the dobot
    def dobotConnect(self):
        if(self.connected):
            print("You're already connected")
        else:
            state = dType.ConnectDobot(self.api, "", 115200)[0]
            if(state == dType.DobotConnect.DobotConnect_NoError):
                print("Connect status:",CON_STR[state])
                self.clear_alarms()
                dType.reset_queue(self.api) 
                home_paramas_status,_,error_type=dType.SetHOMEParams_robust(self.api, self.homeX, self.homeY, self.homeZ, 0, isQueued = 1)
                if home_paramas_status==False:
                    if error_type==1:
                        print("Error: PortOccupied when setting home params")
                    elif error_type==2:
                        print("Error: BufferFull when setting home params")
                    elif error_type==3:
                        print("Error: Timeout when setting home params")
                dType.SetPTPJointParams(self.api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
                dType.SetPTPCommonParams(self.api, 100, 100, isQueued = 1)

                #dType.SetHOMECmd(self.api, temp = 0, isQueued = 1)
                self.connected = True
                self.clear_alarms()
                return self.connected
            else:
                print("Unable to connect")
                print("Connect status:",CON_STR[state])
                return self.connected

    def dobotDisconnect(self):
        if not self.connected or self.api is None:
            print("Not connected to DoBot, cannot disconnect.")
            return

        # Ensure command queue is running
        try:
            dType.SetQueuedCmdStartExec(self.api)
        except Exception as e:
            print("Failed to start command queue:", e)

        # Move to home position and wait for completion
        self.moveHome()

        # Stop the command queue after moving home
        try:
            dType.SetQueuedCmdStopExec(self.api)
        except Exception as e:
            print("Failed to stop command queue:", e)

        # Disconnect
        dType.DisconnectDobot(self.api)
        self.connected = False

    #Delays commands
    def commandDelay(self, lastIndex, timeout_ms=10000):
        dType.SetQueuedCmdStartExec(self.api)
        start = time.time()
        while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            if time.time() - start > timeout_ms:
                dType.SetQueuedCmdStopExec(self.api)
                raise TimeoutError("Queue stalled")
            dType.dSleep(200)
        dType.SetQueuedCmdStopExec(self.api)



    #Toggles suction peripheral on/off
    def toggleSuction(self):
        lastIndex = 0
        if(self.suction):
            lastIndex = dType.SetEndEffectorSuctionCup( self.api, False, False, isQueued = 0)[0]
            self.suction = False
        else:
            lastIndex = dType.SetEndEffectorSuctionCup(self.api, True, True, isQueued = 0)[0]
            self.suction = True
        self.commandDelay(lastIndex)

    #Moves arm to X/Y Location
    def moveArmXY(self,x,y):
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, self.homeZ, 0)[0]
        self.commandDelay(lastIndex)
    #Moves arm to X/Y/Z Location
    def moveArm(self,x,y,z=0,r=0):
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, 0)[0]
        self.commandDelay(lastIndex)
    #Returns to home location
    def moveHome(self):
        self.clear_alarms()
        dType.SetQueuedCmdClear(self.api)
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, self.homeX, self.homeY, self.homeZ, 0)[0]
        self.commandDelay(lastIndex)

        print(f"Moved to home: {self.homeX,self.homeY,self.homeZ,0}")

    #Toggles between hover and item level
    def pickToggle(self, itemHeight):
        lastIndex = 0
        positions = dType.GetPose(self.api)
        if(self.picking):
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, positions[0], positions[1], self.homeZ, 0)[0]
            self.picking = False
        else:
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, positions[0], positions[1], itemHeight, 0)[0]
            self.picking = True
        self.commandDelay(lastIndex)

    # Return the current position of the dobot [x, y, z, rHead, joint1Angle, joint2Angle, joint3Angle, joint4Angle]
    def cur_location(self):
        return dType.GetPose(self.api)
        
 
    def testPIRSensor(self):
        dType.SetIOMultiplexing(self.api, 15, 3, 0)
        timeout_s = 10  # seconds
        start_time = time.time()
        while True:
            PIR_status = dType.GetIODI(self.api, 15)
            if PIR_status[0] == 0:
                print("Object Detected")
                return
            print(PIR_status)
            if time.time() - start_time > timeout_s:
                print("Timeout: No object detected within", timeout_s, "seconds.")
                break
            time.sleep(0.2)
        



    def clear_alarms(self,print_states=False):
        """
        Reset any safety or error alarms that have tripped on the robot.
        Print alarm state before clearing.
        """
        if print_states:
            try:
                alarms = dType.GetAlarmsState(self.api)
                # alarms is usually a tuple: (bytearray, count)
                if alarms and hasattr(alarms[0], '__getitem__'):
                    alarm_bytes = alarms[0]
                    active = [i for i, b in enumerate(alarm_bytes) if b != 0]
                    if active:
                        print(f"[ALARM STATE BEFORE CLEAR] Active alarm bytes at positions: {active}")
                    else:
                        print("[ALARM STATE BEFORE CLEAR] No active alarms.")
                else:
                    print(f"[ALARM STATE BEFORE CLEAR] {alarms}")
            except Exception as e:
                print(f"[ALARM STATE BEFORE CLEAR] Could not retrieve alarms: {e}")
        dType.ClearAllAlarmsState(self.api)
        print(f"[CLEAR] Alarms")
    def _wait(self, lastIndex):
        """Block until the indexed queued command completes."""
        self.commandDelay(lastIndex)

    def set_gripper(self, enable: bool, closed: bool):
        """
        Control the gripper:
          enable=True, closed=False → open & power on
          enable=True, closed=True  → close & power on
          enable=False, closed=False→ power off
        """
        last = dType.SetEndEffectorGripper(
            self.api,
            enable,  # enableCtrl
            closed,  # on/off
            isQueued=1
        )[0]
        self._wait(last)
        time.sleep(0.5)

    def pick_and_place(
        self,
        x_pick: float, y_pick: float, z_pick: float,
        x_place: float, y_place: float, z_place: float
    ):
        
        safe_z = self.homeZ

        # 1) Power-on & open gripper
        self.set_gripper(enable=True, closed=False)
        time.sleep(0.1)

        # 2) Move above pick
        last = dType.SetPTPCmd(
            self.api, dType.PTPMode.PTPMOVLXYZMode,
            x_pick, y_pick, safe_z, 0,
            isQueued=1
        )[0]
        self._wait(last)

        # 3) Descend to pick height
        last = dType.SetPTPCmd(
            self.api, dType.PTPMode.PTPMOVLXYZMode,
            x_pick, y_pick, z_pick, 0,
            isQueued=1
        )[0]
        self._wait(last)

        # 4) Close gripper (grab)
        self.set_gripper(enable=True, closed=True)
        time.sleep(0.1)

        # 5) Lift back up
        last = dType.SetPTPCmd(
            self.api, dType.PTPMode.PTPMOVLXYZMode,
            x_pick, y_pick, safe_z, 0,
            isQueued=1
        )[0]
        self._wait(last)

        # 6) Move above place
        last = dType.SetPTPCmd(
            self.api, dType.PTPMode.PTPMOVLXYZMode,
            x_place, y_place, safe_z, 0,
            isQueued=1
        )[0]
        self._wait(last)

        # 7) Descend to place height
        last = dType.SetPTPCmd(
            self.api, dType.PTPMode.PTPMOVLXYZMode,
            x_place, y_place, z_place, 0,
            isQueued=1
        )[0]
        self._wait(last)

        # 8) Open gripper (release)
        self.set_gripper(enable=True, closed=False)
        time.sleep(0.5)

        # 9) Lift back up
        last = dType.SetPTPCmd(
            self.api, dType.PTPMode.PTPMOVLXYZMode,
            x_place, y_place, safe_z, 0,
            isQueued=1
        )[0]
        self._wait(last)

        self.set_gripper(enable=False, closed=False)
        # time.sleep(0.1)
    def pick_and_place_tray_loc(
        self,
        x_pick: float, y_pick: float, z_pick: float,
        x_place: float, y_place: float, z_place: float,
        staging: tuple = None,          # e.g., (0.0, 180.0, None)  -> z will default to homeZ
        tray_approach: tuple = None     # e.g., (80.0, -260.0, None) -> z will default to homeZ
        ):
            """
            Pick at (x_pick, y_pick, z_pick) -> go to STAGING -> go to TRAY_APPROACH ->
            descend to (x_place, y_place, z_place) -> release -> lift to TRAY_APPROACH.

            staging:      (xs, ys, zs or None)  - safe 'from anywhere' hover point
            tray_approach:(xa, ya, za or None)  - hover above the tray before/after drop
            If zs/za are None, we use self.homeZ for that waypoint's Z.
            """
            safe_z = self.homeZ

            # Resolve waypoints (fill Z with homeZ if omitted)
            if staging is None:
                staging = (0.0, 180.0, safe_z)       # <- pick a globally safe hover you like
            else:
                sx, sy, sz = staging
                staging = (sx, sy, safe_z if sz is None else sz)

            if tray_approach is None:
                tray_approach = (x_place, y_place, safe_z)  # default: hover right above drop XY at safe_z
            else:
                ax, ay, az = tray_approach
                tray_approach = (ax, ay, safe_z if az is None else az)

            sx, sy, sz = staging
            ax, ay, az = tray_approach

            # 1) Power-on & open gripper
            self.set_gripper(enable=True, closed=False)
            time.sleep(0.1)

            # =======================
            # PICK SEQUENCE
            # =======================

            # 2) Move above pick (vertical approach)
            last = dType.SetPTPCmd(
                self.api, dType.PTPMode.PTPMOVLXYZMode,
                x_pick, y_pick, safe_z, 0,
                isQueued=1
            )[0]
            self._wait(last)

            # 3) Descend to pick height
            last = dType.SetPTPCmd(
                self.api, dType.PTPMode.PTPMOVLXYZMode,
                x_pick, y_pick, z_pick, 0,
                isQueued=1
            )[0]
            self._wait(last)

            # 4) Close gripper (grab)
            self.set_gripper(enable=True, closed=True)
            time.sleep(0.1)

            # # 5) Lift back up to safe_z
            # last = dType.SetPTPCmd(
            #     self.api, dType.PTPMode.PTPMOVLXYZMode,
            #     x_pick, y_pick, safe_z, 0,
            #     isQueued=1
            # )[0]
            # self._wait(last)


            # 6) Move to STAGING (joint interpolation is more robust for long XY hops)
            last = dType.SetPTPCmd(
                self.api, dType.PTPMode.PTPMOVJXYZMode,
                sx, sy, sz, 0,
                isQueued=1
            )[0]
            self._wait(last)

            # 7) Move to TRAY_APPROACH (joint or linear at hover height)
            last = dType.SetPTPCmd(
                self.api, dType.PTPMode.PTPMOVJXYZMode,
                ax, ay, az, 0,
                isQueued=1
            )[0]
            self._wait(last)

            # =======================
            # DROP SEQUENCE
            # =======================

            # 8) Move horizontally above final drop XY at hover height (keep az)
            last = dType.SetPTPCmd(
                self.api, dType.PTPMode.PTPMOVLXYZMode,
                x_place, y_place, az, 0,
                isQueued=1
            )[0]
            self._wait(last)

            # 9) Descend to exact drop height
            last = dType.SetPTPCmd(
                self.api, dType.PTPMode.PTPMOVLXYZMode,
                x_place, y_place, z_place, 0,
                isQueued=1
            )[0]
            self._wait(last)

            # 10) Open gripper (release)
            self.set_gripper(enable=True, closed=False)
            time.sleep(0.5)

            # 11) Lift back to TRAY_APPROACH height
            last = dType.SetPTPCmd(
                self.api, dType.PTPMode.PTPMOVLXYZMode,
                x_place, y_place, az, 0,
                isQueued=1
            )[0]
            self._wait(last)

            # 12) Power off gripper (optional)
            self.set_gripper(enable=False, closed=False)
            # 6) Move to STAGING (joint interpolation is more robust for long XY hops)
            last = dType.SetPTPCmd(
                self.api, dType.PTPMode.PTPMOVJXYZMode,
                sx, sy, sz, 0,
                isQueued=1
            )[0]
            self._wait(last)
            
    def set_conveyor_speed(self, mm_per_s: float, *, diameter_mm: float = 32.0,
                           step_deg: float = 1.8, microstep: int = 16, gear_ratio: float = 5.0,
                           queue: bool = True,reverse=False):
        """
        Set conveyor linear speed in mm/s. Use 0 to stop.
        diameter_mm: roller diameter
        step_deg: motor step angle (degrees)
        microstep: microstepping factor (e.g., 16 for 1/16)
        gear_ratio: motor_rev : roller_rev (e.g., 5:1)
        """
        # Convert linear speed to pulse rate (pulses/s)
        steps_per_motor_rev = 360.0 / step_deg
        steps_per_roller_rev = steps_per_motor_rev * microstep * gear_ratio
        mm_per_roller_rev = 3.1415926535898 * diameter_mm
        pulses_per_mm = steps_per_roller_rev / mm_per_roller_rev
        speed_pps = int(abs(mm_per_s) * pulses_per_mm)
        if reverse:
            speed_pps*=-1
    
        if mm_per_s == 0:
            # Stop & disable
            dType.SetEMotor(api=self.api, index=0, isEnabled=0, speed=0, isQueued=queue)
        else:
            # Enable & run
            dType.SetEMotor(api=self.api, index=0, isEnabled=1, speed=speed_pps, isQueued=queue)
    

# Some new functions 
    def set_speed_ratio(self, velocity: int, acceleration: int):
        """Set global PTP speed/accel ratios (0..100)."""
        dType.SetPTPCommonParams(self.api, int(velocity), int(acceleration), isQueued=1)
        # wait for the queued param change to apply
        dType.SetQueuedCmdStartExec(self.api)
        dType.SetQueuedCmdStopExec(self.api)

    def move_to_pose(self, x: float, y: float, z: float, r: float):
        """Absolute cartesian move with rotation kept (MOVL)."""
        last = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, r, isQueued=1)[0]
        self.commandDelay(last)

    def estop(self):
        """Immediately stop queued motion & clear alarms."""
        # Force-stop the queue immediately and clear alarms so user can Home again
        dType.SetQueuedCmdForceStopExec(self.api)
        self.clear_alarms()
    
    def pause_for_inspection(self):
        """
        Soft-pause the queue so no NEW commands are consumed.
        The currently executing command is allowed to finish naturally.
        Returns a small snapshot dict so you can inspect state.
        """
        # Stop consuming queue entries (soft stop)
        dType.SetQueuedCmdStopExec(self.api)  # soft pause
        # Read current index & pose for your logs/inspection
        cur_idx = dType.GetQueuedCmdCurrentIndex(self.api)[0]
        pose = dType.GetPose(self.api)  # [x, y, z, rHead, j1, j2, j3, j4]
        snap = {"current_index": cur_idx, "pose": pose}
        print("[PAUSE] snapshot:", snap)
        return snap
