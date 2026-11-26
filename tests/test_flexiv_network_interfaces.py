"""Connection routing regression tests; no robot connections or motion."""

import sys
import unittest
from unittest.mock import MagicMock, patch

# The routing tests replace vendor calls and run without the Flexiv SDK.
with patch.dict(sys.modules, {"flexivrdk": MagicMock()}):
    from ImplicitRDP.real_world.robot import single_flexiv_controller as controller_module


class FlexivNetworkInterfaceTests(unittest.TestCase):
    def make_controller(self, enable_tdk, interfaces):
        # Bypass hardware initialization; exercise the connection methods only.
        controller = controller_module.FlexivController.__new__(controller_module.FlexivController)
        controller.enable_tdk = enable_tdk
        controller.leader_sn = "LEADER-TEST"
        controller.network_interface_whitelist = interfaces
        controller.mode = MagicMock()
        return controller

    @patch.object(controller_module.flexivrdk, "Robot")
    def test_rdk_uses_configured_interfaces(self, robot):
        for interfaces in ([], ["192.168.1.65"]):
            with self.subTest(interfaces=interfaces):
                controller = self.make_controller(False, interfaces)
                self.assertIs(controller._create_robot_handle("FOLLOWER-TEST"), robot.return_value)
                robot.assert_called_with("FOLLOWER-TEST", interfaces)

    def test_tdk_uses_same_interfaces_as_follower_preparation(self):
        controller = self.make_controller(True, ["192.168.1.65"])
        tdk = MagicMock()
        leader, follower = MagicMock(), MagicMock()
        tdk.TransparentCartesianTeleopLAN.return_value.instances.return_value = (leader, follower)
        with patch.dict("sys.modules", {"flexivtdk": tdk}), patch.object(
            controller, "_set_rotation_via_rdk"
        ) as prepare:
            self.assertIs(controller._create_robot_handle("FOLLOWER-TEST"), follower)
        prepare.assert_called_once_with("FOLLOWER-TEST")
        tdk.TransparentCartesianTeleopLAN.assert_called_once_with(
            [("LEADER-TEST", "FOLLOWER-TEST")], ["192.168.1.65"]
        )

    @patch.object(controller_module.time, "sleep")
    @patch.object(controller_module.flexivrdk, "Robot")
    def test_preparation_uses_configured_interfaces(self, robot, sleep):
        controller = self.make_controller(True, ["192.168.1.65"])
        robot.return_value.states.return_value.tcp_pose = [0.0] * 7
        with patch.object(controller, "_check_and_enable_robot"), patch.object(
            controller, "parse_pt_terminated_or_reachedTarget", return_value=True
        ):
            controller._set_rotation_via_rdk("FOLLOWER-TEST")
        robot.assert_called_once_with("FOLLOWER-TEST", ["192.168.1.65"])


if __name__ == "__main__":
    unittest.main()
