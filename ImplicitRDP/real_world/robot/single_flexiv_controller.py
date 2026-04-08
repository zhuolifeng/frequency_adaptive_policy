import flexivrdk

import threading
import time
from typing import List, Optional
from loguru import logger


class DummyGripper():
    class DummyGripperStates():
        def __init__(self):
            self.width = 0.10
            self.force = 0.0
    
    def __init__(self):
        self._states = self.DummyGripperStates()

    def states(self):
        return self._states

    def Grasp(self, force_limit: float):
        self._states.force = force_limit
        logger.info(f"DummyGripper: Grasping with force limit {force_limit}")

    def Move(self, width: float, velocity: float, force_limit: float):
        self._states.width = width
        self._states.force = force_limit
        logger.info(f"DummyGripper: Moving to width {width} with velocity {velocity} and force limit {force_limit}")
    
    def Stop(self):
        self._states.force = 0.0
        logger.info(f"DummyGripper: Stopping")

class FlexivController():
    def __init__(self,
                 robot_serial_number,
                 gripper_name: Optional[str] = None,
                 enable_tdk: bool = False,
                 leader_sn: Optional[str] = None,
                 pedal_poll_interval: float = 0.01,
                 ) -> None:
        self.DOF=7
        self.is_default_cartesian_impedance = True
        self.enable_tdk = enable_tdk
        self.leader_sn = leader_sn
        self.pedal_poll_interval = pedal_poll_interval
        self.teleop = None
        self.flexivtdk = None
        self._leader_robot = None
        self.teleop_engaged = False
        self.tdk_active = False
        self.pedal_thread: Optional[threading.Thread] = None
        self._stop_pedal_event = threading.Event()

        try:
            self.mode = flexivrdk.Mode
            self.robot = self._create_robot_handle(robot_serial_number)
            self.nominal_stiffness = self.robot.info().K_x_nom
            self.gripper = flexivrdk.Gripper(self.robot)

            self.clear_fault()
            logger.info("Enabling robot ...")
            self._check_and_enable_robot(self.robot, "robot")
            if gripper_name is not None:
                logger.info(f"Enabling gripper {gripper_name} ...")
                self.gripper.Enable(gripper_name)
            else:
                logger.info("No gripper name provided, using dummy gripper ...")
                self.gripper = DummyGripper()

            seconds_waited = 0
            while not self.robot.operational():
                time.sleep(1)
                seconds_waited += 1
                if seconds_waited == 10:
                    logger.warning(
                        "Still waiting for robot to become operational, please check that the robot 1) "
                        "has no fault, 2) is in [Auto (remote)] mode")
            logger.info("Robot is now operational")
            # important:reset the FTSensor in order to set the mode to NRT_CARTESIAN_MOTION_FORCE for RIZON 4s
            if self.enable_tdk and self.teleop is not None:
                self._start_tdk()
            else:
                # Reset the FT sensor before entering cartesian control.
                self.execute_primitive("ZeroFTSensor", {})
                self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        except Exception as e:
            logger.error("Error occurred while connecting to robot server: %s" % str(e))
            return None

    def _create_robot_handle(self, robot_serial_number: str) -> Any:
        if not self.enable_tdk:
            logger.info(f"Creating RDK robot interface for follower {robot_serial_number}")
            return flexivrdk.Robot(robot_serial_number)

        if self.leader_sn is None:
            raise ValueError("leader_sn must be specified when enable_tdk=True")

        try:
            import flexivtdk
            self.flexivtdk = flexivtdk
        except ImportError as e:
            raise ImportError(f"flexivtdk not available: {e}. Install with: pip install flexivtdk") from e

        self._set_rotation_via_rdk(robot_serial_number)

        logger.info(
            f"Creating TDK teleop instance (leader={self.leader_sn}, follower={robot_serial_number})"
        )
        self.teleop = self.flexivtdk.TransparentCartesianTeleopLAN(
            [(self.leader_sn, robot_serial_number)], []
        )
        self._leader_robot, follower_robot = self.teleop.instances(0)
        logger.info("TDK teleop instance created; follower robot handle obtained from teleop.instances(0)")
        return follower_robot

    def _set_rotation_via_rdk(self, robot_serial_number: str) -> None:
        """Temporarily connect to a follower arm via RDK, move to a fixed orientation, then release."""
        try:
            logger.info(f"Connecting to follower arm {robot_serial_number} via RDK to zero FT sensor ...")
            follower_tmp = flexivrdk.Robot(robot_serial_number)
            self._check_and_enable_robot(follower_tmp, robot_serial_number)
            self.execute_primitive("ZeroFTSensor", {})
            current_tcp = follower_tmp.states().tcp_pose
            fixed_orientation = [0.17364820837974548, 2.201380311817047e-06, 0.9848077297210693, 8.990650144369283e-07]
            target_tcp = current_tcp[:3] + fixed_orientation
            follower_tmp.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
            follower_tmp.SendCartesianMotionForce(
                target_tcp,
                [0.0] * 6,
                [0.0] * 6,
                0.5,
                1.0,
            )
            logger.info(f"Move follower arm {robot_serial_number} to a fixed orientation")
            del follower_tmp
            time.sleep(1.0)
        except Exception as e:
            logger.error(f"Failed to zero FT sensor on robot {robot_serial_number}: {e}")

    def _start_tdk(self) -> None:
        if self.teleop.stopped(0):
            logger.info("Initializing TDK teleop ...")
            self.teleop.Init()
            self.teleop.Start()
            self.tdk_active = True
            logger.info("TDK teleop started")
        else:
            self.tdk_active = True


        self._lock_tdk_rotation_in_world()
        self._start_pedal_thread()

    def _lock_tdk_rotation_in_world(self) -> None:
        if self.teleop is None or self.flexivtdk is None:
            return

        try:
            cmd = self.teleop.GetAxisLockState(0)
        except Exception:
            cmd = self.flexivtdk.AxisLock()
            cmd.lock_trans_axis = [False, False, False]
            cmd.lock_ori_axis = [False, False, False]

        cmd.lock_trans_axis = [False, False, False]
        cmd.lock_ori_axis = [True, True, True]
        cmd.coord = self.flexivtdk.CoordType.WORLD
        self.teleop.SetAxisLockCmd(0, cmd)
        time.sleep(1.0)
        logger.info("Locked all TDK rotation axes in WORLD coordinates")

    def _start_pedal_thread(self) -> None:
        if self.pedal_thread and self.pedal_thread.is_alive():
            return

        logger.info("Starting pedal thread ...")
        self._stop_pedal_event.clear()
        self.pedal_thread = threading.Thread(
            target=self._read_digital_input_task,
            name="flexiv_tdk_pedal",
            daemon=True,
        )
        self.pedal_thread.start()
        logger.info("Pedal thread started")

    def _set_teleop_engaged(self, engaged: bool) -> None:
        if self.teleop is None or self.teleop_engaged == engaged:
            return

        self.teleop.Engage(0, engaged)
        self.teleop_engaged = engaged
        logger.info(f"Teleop {'engaged' if engaged else 'disengaged'}")

    def _read_digital_input_task(self) -> None:
        """Poll the leader arm pedal and engage/disengage teleop on demand."""
        while not self._stop_pedal_event.is_set():
            try:
                if self.teleop is None:
                    time.sleep(self.pedal_poll_interval)
                    continue

                leader_di, _ = self.teleop.digital_inputs(0)
                engage_state = bool(leader_di and len(leader_di) > 0 and leader_di[0])
                if not self.tdk_active:
                    time.sleep(self.pedal_poll_interval)
                    continue
                if engage_state != self.teleop_engaged:
                    pedal_state = "pressed" if engage_state else "released"
                    action = "engaging" if engage_state else "disengaging"
                    logger.info(f"Pedal {pedal_state}, {action} teleop")
                    self._set_teleop_engaged(engage_state)
            except Exception as e:
                logger.error(f"Error in pedal polling task: {e}")
            time.sleep(self.pedal_poll_interval)

    def close(self) -> None:
        self._stop_pedal_event.set()
        if self.pedal_thread and self.pedal_thread.is_alive():
            self.pedal_thread.join(timeout=1.0)
        self.pedal_thread = None

        if self.teleop is not None:
            try:
                if not self.teleop.stopped(0):
                    try:
                        self._set_teleop_engaged(False)
                    except Exception:
                        self.teleop_engaged = False
                    self.teleop.Stop()
                    logger.info("TDK teleop stopped")
            except Exception as e:
                logger.error(f"Error stopping TDK teleop: {e}")
            finally:
                self.tdk_active = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _check_and_enable_robot(self, robot, name: str, timeout: int = 30) -> None:
        if robot.fault():
            logger.warning(f"Fault occurred on {name}, trying to clear ...")
            robot.ClearFault()
            time.sleep(2)
            if robot.fault():
                raise RuntimeError(f"Fault on {name} cannot be cleared")
            logger.info(f"Fault on {name} is cleared")

        logger.info(f"Enabling {name} ...")
        robot.Enable()

        seconds_waited = 0
        while not robot.operational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                logger.warning(
                    f"Still waiting for {name} to become operational, please check that the robot 1) "
                    "has no fault, 2) is in [Auto (remote)] mode"
                )
            if seconds_waited >= timeout:
                raise TimeoutError(f"Timed out waiting for {name} to become operational")
        logger.info(f"{name.capitalize()} is now operational")


    def clear_fault(self):
        # Fault Clearing
        # ==========================================================================================
        # Check if the robot has fault
        if self.robot.fault():
            logger.warning("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            self.robot.ClearFault()
            time.sleep(2)
            # Check again
            if self.robot.fault():
                logger.error("Fault cannot be cleared, exiting ...")
                return
            logger.info("Fault on robot server is cleared")

    def get_current_robot_states(self) -> object:
        return type('obj', (object,), {
            'tcp_pose': self.robot.states().tcp_pose,
            'tcp_vel': self.robot.states().tcp_vel,
            'ext_wrench_in_tcp': self.robot.states().ext_wrench_in_tcp,
        })

    def get_current_gripper_states(self) -> flexivrdk.GripperStates:
        return self.gripper.states()
    
    def is_using_default_cartesian_impedance(self) -> bool:
        return self.is_default_cartesian_impedance

    def get_current_gripper_force(self) -> float:
        return self.gripper.states().force

    def get_current_gripper_width(self) -> float:
        return self.gripper.states().width

    def get_current_q(self) -> List[float]:
        # Return current joint values of the robot arm under flexivAPI
        return self.robot.states().q

    def get_current_tcp(self) -> List[float]:
        # Return current TCP pose of the robot arm under flexivAPI
        return self.robot.states().tcp_pose

    def move(self, target_q):
        v = [1.5]*self.DOF # velocity limit
        a = [0.8]*self.DOF # acceleration limit
        self.robot.SendJointPosition(
                target_q,
                [0.0]*self.DOF,
                [0.0]*self.DOF,
                v,
                a)
            
    def tcp_move(self, target_tcp):
        self.robot.SendCartesianMotionForce(
                target_tcp, 
                [0.0]*6, 
                0.5,
                1.0)

    def execute_primitive(self, primitive_name: str, input_params: dict = {}):
        if self.enable_tdk:
            logger.info(
                f"execute_primitive({primitive_name}) requested while TDK is enabled; "
                "skip RDK primitive execution to avoid control-mode mismatch"
            )
            return
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive(primitive_name, input_params)
        while not self.parse_pt_terminated_or_reachedTarget(self.robot.primitive_states()):
            time.sleep(0.001)
        if not self.enable_tdk:
            self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

    @staticmethod
    def parse_pt_terminated_or_reachedTarget(pt_states):
        result = False
        if "terminated" in pt_states:
            result |= pt_states["terminated"] == 1
        if "reachedTarget" in pt_states:
            result |= pt_states["reachedTarget"] == 1
        return result

    def get_nominal_stiffness(self):
        return self.nominal_stiffness

    def set_cartesian_impedance(self, stiffness, damping):
        start_time = time.time()
        self.robot.SetCartesianImpedance(stiffness, damping)
        self.is_default_cartesian_impedance = stiffness == self.nominal_stiffness and damping == [0.7]*6
        logger.debug(f"Taken {time.time() - start_time:.4f} seconds to set stiffness: {stiffness} and damping: {damping}")

    def reset_cartesian_impedance(self):
        start_time = time.time()
        self.robot.SetCartesianImpedance(self.nominal_stiffness)
        self.is_default_cartesian_impedance = True
        logger.debug(f"Taken {time.time() - start_time:.4f} seconds to reset stiffness to nominal value: {self.nominal_stiffness}")