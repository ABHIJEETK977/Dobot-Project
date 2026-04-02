#!/usr/bin/env python

import sys
sys.path.insert(1,'./DLL')
import DobotDllType as dType
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
    def __init__(self, homeX, homeY, homeZ):
        self.suction = False
        self.picking = False
        self.api = dType.load()
        self.homeX = homeX
        self.homeY = homeY
        self.homeZ = homeZ
        self.connected = False
        self.dobotConnect()

    def __del__(self):
        self.dobotDisconnect()

    #Attempts to connect to the dobot
    def dobotConnect(self):
        if(self.connected):
            print("You're already connected")
        else:
            state = dType.ConnectDobot(self.api, "", 115200)[0]
            if(state == dType.DobotConnect.DobotConnect_NoError):
                print("Connect status:",CON_STR[state])
                dType.SetQueuedCmdClear(self.api)

                dType.SetHOMEParams(self.api, self.homeX, self.homeY, self.homeZ, 0, isQueued = 1)
                dType.SetPTPJointParams(self.api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
                dType.SetPTPCommonParams(self.api, 100, 100, isQueued = 1)

                #dType.SetHOMECmd(self.api, temp = 0, isQueued = 1)
                self.connected = True
                self.clear_alarms()
                self.moveHome()
                return self.connected
            else:
                print("Unable to connect")
                print("Connect status:",CON_STR[state])
                return self.connected

    #Returns to home location and then disconnects
    def dobotDisconnect(self):
        self.moveHome()
        dType.DisconnectDobot(self.api)

    #Delays commands
    def commandDelay(self, lastIndex):
        dType.SetQueuedCmdStartExec(self.api)
        while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
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

    #Moves arm to X/Y/Z Location
    def moveArmXY(self,x,y):
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, self.homeZ, 0)[0]
        self.commandDelay(lastIndex)

    #Returns to home location
    def moveHome(self):
        self.clear_alarms()
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, self.homeX, self.homeY, self.homeZ, 0)[0]
        self.commandDelay(lastIndex)

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

    def testHandHoldTeaching(self):        
    
        STEP_PER_CIRCLE = 360.0 / 1.8 * 5.0 * 16.0
        MM_PER_CIRCLE = 3.1415926535898 * 32.0
        vel = float(10.0) * STEP_PER_CIRCLE / MM_PER_CIRCLE # speed pulses required for a conveyor speed of 10.0[mm/s]
        #vel = float(0) * STEP_PER_CIRCLE / MM_PER_CIRCLE # speed pulses required for a conveyor speed of 0[mm/s]
        #dType.SetEMotor(self.api, 0, 1, int(vel), True)
        dType.SetEMotor(self.api, 0, 0, 1600, True)
        dType.SetEMotor(self.api, 0, 1, 1600, True)
        dType.dSleep(5000)

    def testPIRSensor(self):
        dType.SetIOMultiplexing(self.api, 15, 3, 1)
        while (True):
            PIR_status = dType.GetIODI(self.api, 15)
            print(PIR_status)
            dType.dSleep(500) 

    def cur_location(self):
        return dType.GetPose(self.api)
    
    def clear_alarms(self):
        """
        Reset any safety or error alarms that have tripped on the robot.

        """
        dType.ClearAllAlarmsState(self.api)
        print(f"[CLEAR] Alarms")
    def _wait(self, lastIndex):
        """Block until the indexed queued command completes."""
        self.commandDelay(lastIndex)

    def set_gripper(self, enable: bool, closed: bool):
        """
        Control the gripper:
          enable=True, closed=False ? open & power on
          enable=True, closed=True  ? close & power on
          enable=False, closed=False? power off
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