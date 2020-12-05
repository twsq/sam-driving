import numpy as np
import scipy
import sys
import os
import glob
import torch
import datetime

from scipy.misc import imresize
from PIL import Image

import matplotlib.pyplot as plt

try:
    from carla08 import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

from carla08.agent import CommandFollower
from carla08.client import VehicleControl

from network import CoILModel
from configs import g_conf
from logger import coil_logger

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class CoILAgent(object):

    def __init__(self, checkpoint, town_name, carla_version='0.84'):

        # Set the carla version that is going to be used by the interface
        self._carla_version = carla_version
        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        # Create model
        self._model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        self.first_iter = True
        # Load the model and prepare set it for evaluation
        self._model.load_state_dict(checkpoint['state_dict'])
        self._model.cuda()
        self._model.eval()

        # If we are evaluating squeeze model (so we are using ground truth seg mask), 
        # also run the autopilot to get its stop intentions
        if g_conf.USE_ORACLE or g_conf.USE_FULL_ORACLE or "seg" in g_conf.SENSORS.keys():
            self.control_agent = CommandFollower(town_name)

    def run_step(self, measurements, sensor_data, directions, target):
        """
            Run a step on the benchmark simulation
        Args:
            measurements: All the float measurements from CARLA ( Just speed is used)
            sensor_data: All the sensor data used on this benchmark
            directions: The directions, high level commands
            target: Final objective. Not used when the agent is predicting all outputs.

        Returns:
            Controls for the vehicle on the CARLA simulator.

        """
        # Get speed and high-level turning command
        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = measurements.player_measurements.forward_speed / g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([directions])

        # If we're evaluating squeeze network (so we are using ground truth seg mask)
        if "seg" in g_conf.SENSORS.keys():
            # Run the autopilot agent to get stop intentions
            _, state = self.control_agent.run_step(measurements, [], [], target)
            inputs_vec = []
            for input_name in g_conf.INTENTIONS:
                inputs_vec.append(float(state[input_name]))
            intentions = torch.cuda.FloatTensor(inputs_vec).unsqueeze(0)
            # Run squeeze network
            model_outputs = self._model.forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                       directions_tensor, intentions, benchmark=True)
        else:
            # Run driving model
            model_outputs = self._model.forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                       directions_tensor, benchmark=True)

        steer, throttle, brake = self._process_model_outputs(model_outputs[0])
        if self._carla_version == '0.9':
            import carla
            control = carla.VehicleControl()
        else:
            control = VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        # There is the posibility to replace some of the predictions with oracle predictions.
        if g_conf.USE_ORACLE:
            _, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, target)

        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(steer)},
                                    self.checkpoint['iteration'])
        self.first_iter = False
        
        return control

    def _process_sensors(self, sensors):

        iteration = 0
        sensor_dict = {}
        for name, size in g_conf.SENSORS.items():

            if self._carla_version == '0.9':
                sensor = sensors[name][g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]
            else:
                sensor = sensors[name].data[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]

            # Process RGB image or CARLA seg mask
            if name == 'rgb':
                # Resize image, convert it to [0, 1] BGR image
                sensor = scipy.misc.imresize(sensor, (size[1], size[2]))
                sensor = sensor[:, :, ::-1]
                sensor = np.swapaxes(sensor, 0, 1)
                sensor = np.transpose(sensor, (2, 1, 0))
                sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor)
            elif name == 'seg':
                seg = scipy.misc.imresize(sensor, (size[1], size[2]), 'nearest')
                # Re-map classes, mapping irrelevant classes to a "nuisance" class
                class_map = \
                    {0: 0, # None
                     1: 0, # Buildings -> None
                     2: 0, # Fences -> None
                     3: 0, # Other -> None
                     4: 1, # Pedestrians kept
                     5: 0, # Poles -> None
                     6: 2, # RoadLines kept
                     7: 3, # Roads kept
                     8: 2, # Sidewalks mapped to roadlines (both are boundaries of road)
                     9 : 0, # Vegetation -> None
                     10: 4, # Vehicles kept
                     11: 0, # Walls -> None
                     12: 5} # TrafficSigns kept (for traffic lights)
                new_seg = np.zeros((seg.shape[0], seg.shape[1]))
                # Remap classes
                for key, value in class_map.items():
                    new_seg[np.where(seg == key)] = value 
                # One hot encode seg mask, for now hardcode max of class map values + 1
                new_seg = np.eye(6)[new_seg.astype(np.int32)]
                new_seg = new_seg.transpose(2, 0, 1)
                new_seg = new_seg.astype(np.float)
                sensor = torch.from_numpy(new_seg).type(torch.FloatTensor)

            sensor = sensor.unsqueeze(0)
            sensor_dict[name] = sensor

        return sensor_dict

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0


        return steer, throttle, brake

    def _get_oracle_prediction(self, measurements, target):
        # For the oracle, the current version of sensor data is not really relevant.
        control, _ = self.control_agent.run_step(measurements, [], [], target)

        return control.steer, control.throttle, control.brake
