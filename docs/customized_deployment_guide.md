# Deploy on a robot

The reference setup uses a Flexiv Rizon arm with a six-axis force/torque sensor, an external USB camera and a wrist camera. The dual-switch configuration uses an iPhone with Record3D on the wrist. Other tasks include RealSense and tactile sensors. [Reference hardware](../assets/hardware_settings.png)

## Prepare the workstation

Install ROS 2 Humble on Ubuntu 22.04, source `/opt/ros/humble/setup.bash` and activate the Python 3.10 environment from the README in each terminal. Install the full Python requirements and the Flexiv RDK bindings for your controller. Configurations with `right_enable_tdk` also require the vendor's TDK bindings and a leader robot. The camera launcher imports Record3D and RealSense bindings, and the real-time camera process requests priority 99. Configure the workstation's real-time scheduling permissions and choose CPU IDs available to each process. [ROS 2 installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) · [Camera launcher](../camera_node_launcher.py)

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the robot serial numbers, service addresses and calibration directory for your installation in the robot environment and task YAMLs. Verify the camera assignments, CPU affinity, control mode, action representation and workspace limits for the physical setup. The checked-in calibration matrices describe the reference installation. [Robot environment](../ImplicitRDP/config/task/real_robot_env.yaml) · [Dual-switch hardware configuration](../ImplicitRDP/config/task/real_dual_switch_iphone_usb_camera_30_10fps.yaml)

## Record a demonstration

Start the control service and the camera launcher in separate terminals using the same task. The camera launcher also starts the device mapping service.

```bash
# Terminal 1
python control.py task=real_dual_switch_iphone_usb_camera_30_10fps
```

```bash
# Terminal 2
python camera_node_launcher.py task=real_dual_switch_iphone_usb_camera_30_10fps
```

Start the recorder in a third terminal and perform the demonstration. Stop it with Ctrl+C to save the episode. Give each subsequent episode a new numeric filename. The recorder writes the example below to `data/dual_switch_v1_round_1/0.pkl`.

```bash
python record_data.py --task real_dual_switch_iphone_usb_camera_30_10fps \
    --save_to_disk --save_file_dir dual_switch_v1_round_1 --save_file_name 0.pkl
```

For tasks configured for manual kinematic teaching, the reference single-button keyboard sends the Pause key. Hold it to enable compliance during teaching and release it to end compliance, or select toggle mode in the controller. The dual-switch example above uses an external controller. [Kinematic teaching controller](../ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py) · [Force monitoring](data_collection_tips.md)

## Run a trained policy

With the control and camera services running, start the evaluation in another terminal. Set the workspace and task to match the checkpoint. This example uses the dual-switch 30 Hz iMeanFlow policy and the runner's right-arm 6-DOF action mode.

```bash
CONFIG_NAME=train_dp_30hz_imeanflow_workspace \
    TASK_CONFIG=real_dual_switch_image_wrench_offline_10fps_30hz \
    CKPT_PATH=/path/to/checkpoints/latest.ckpt bash eval.sh \
    task.env_runner.action_type=right_arm_6DOF
```

For a frequency-adaptive checkpoint, use the multi-frequency training workspace and the dual-rate runner for the same task. Set `policy.training_stage` to match the checkpoint, enable `policy.use_confidence_based_head_selection`, and set `policy.confidence_threshold_for_head_selection` from calibration data. The runner chooses the 30 Hz head when the predicted score exceeds the threshold and the 10 Hz head otherwise. [Dual-switch dual-rate runner configuration](../ImplicitRDP/config/task/real_dual_switch_image_wrench_offline_10fps_dual_hz.yaml)

For optional experiment video recording, configure the camera server address in the task file and start `vcamera_server.py` with the matching camera type, ID and port. [MindVision camera setup](../third_party/mvsdk/README.md)

### Add Customized Tasks
There are two kinds of task configs: **shared configs** and **policy-related configs**.
#### Shared Configs
Shared configs define the robot's degrees of freedom, as well as the sensors used and their parameters.
You can take [ImplicitRDP/config/task/real_flip_one_usb_camera_kineteach_10fps.yaml](../ImplicitRDP/config/task/real_flip_one_usb_camera_kineteach_10fps.yaml)
as an example.

Note that the `robot_control_mode` represents the degrees of freedom of the robot action during the kinematic teaching.
You can add new mode in [ImplicitRDP/common/data_models.py](../ImplicitRDP/common/data_models.py) and
modify [ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py](../ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py) accordingly.

#### Policy-related Configs
Policy-related configs define the obs and action shape, env runner parameters, and dataset parameters.

### Add Customized Force Sensors
1. **Implement the Sensor Publisher.**
   Because Flexiv Rizon 4s is equipped with a built-in 6-axis force/torque sensor at the end effector, we have implemented the publisher in [ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py](../ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py).
   If you want to use a separate force sensor, you can add an individual force sensor publisher to publish the wrench to the ROS topic.
2. **Modify the Device Mapping Server.**
   Our experiments use many different settings of sensor hardwares, so we use `Device Mapping Server` as an online database for other processes to query the current hardware settings.
   It provides the mapping from the sensor to the ROS2 topic name,
   which is requested by the `Data Recorder` and `Runner`. Please add the new sensor mapping in [ImplicitRDP/real_world/device_mapping/device_mapping_server.py](../ImplicitRDP/real_world/device_mapping/device_mapping_server.py).
    
3. **Modify Services.**
   Modify the following services to be compatible with the new sensors:
   - Data Recorder: [ImplicitRDP/real_world/teleoperation/data_recorder.py](../ImplicitRDP/real_world/teleoperation/data_recorder.py)
   - Runner:
      - Real Runner: [ImplicitRDP/env_runner/real_runner.py](../ImplicitRDP/env_runner/real_runner.py)
      - Real Stable Reactive Runner: [ImplicitRDP/env_runner/real_stable_reactive_runner.py](../ImplicitRDP/env_runner/real_stable_reactive_runner.py)

### Add Customized Robots
1. **Implement the Robot Server.**
   Note that we change the stiffness of the robot for kinematic teaching, so you need to adopt a robot which supports stiffness adjustment.
   Refer to [ImplicitRDP/real_world/robot/bimanual_flexiv_server.py](../ImplicitRDP/real_world/robot/bimanual_flexiv_server.py)
   and implement a robot server for your robot with the same API.
   This server is requested by the `Kineteach Controller` and `Real Env`.
2. **Implement the Robot Publisher.**
   Refer to [ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py](../ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py)
   and implement a similar publisher for your robot.
3. **(Optional) Modify Services.**
   The following services may need to be modified to be compatible with the new robot:
   - Kineteach Controller: [ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py](../ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py)
   - Real Env: [ImplicitRDP/env/real_bimanual/real_env.py](../ImplicitRDP/env/real_bimanual/real_env.py)
