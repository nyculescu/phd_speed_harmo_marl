import configparser
from sumolib import checkBinary
import os
import sys
import logging
import ast
import traci
import numpy as np
from enum import Enum

def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    try:
        config['gui'] = content['simulation'].getboolean('gui')
        config['total_episodes'] = content['simulation'].getint('total_episodes')
        config['max_steps'] = content['simulation'].getint('max_steps')
        config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
        config['num_layers'] = content['model'].getint('num_layers')
        config['width_layers'] = content['model'].getint('width_layers')
        config['batch_size'] = content['model'].getint('batch_size')
        config['learning_rate'] = content['model'].getfloat('learning_rate')
        config['training_epochs'] = content['model'].getint('training_epochs')
        config['memory_size_min'] = content['memory'].getint('memory_size_min')
        config['memory_size_max'] = content['memory'].getint('memory_size_max')
        config['num_states'] = content['agent'].getint('num_states')
        config['actions'] = ast.literal_eval(content['agent'].get('actions'))
        config['gamma'] = content['agent'].getfloat('gamma')
        config['models_path_name'] = content['dir']['models_path_name']
        config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
        logger.debug("GUI configuration loaded successfully.")
        return config
    except KeyError as e:
        logger.error(f"Failed to load GUI configuration: {e}")
        raise

def import_test_configuration(config_file):
    """
    Read the config file regarding the testing and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    try:
        config['gui'] = content['simulation'].getboolean('gui')
        config['max_steps'] = content['simulation'].getint('max_steps')
        config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
        config['episode_seed'] = content['simulation'].getint('episode_seed')
        config['num_states'] = content['agent'].getint('num_states')
        config['num_actions'] = content['agent'].getint('num_actions')
        config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
        config['models_path_name'] = content['dir']['models_path_name']
        config['model_to_test'] = content['dir'].getint('model_to_test') 
        logger.debug("GUI configuration loaded successfully.")
        return config
    except KeyError as e:
        logger.error(f"Failed to load GUI configuration: {e}")
        raise

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join(f'{os.getcwd()}\\configs\\sumo', sumocfg_file_name), "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd

def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 


def set_test_path(models_path_name, model_n):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_'+str(model_n), '')

    if os.path.isdir(model_folder_path):    
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')

def get_vehicle_speed_at_point(vehicle_id, radar_x, radar_y, tolerance=5):
    """
    Retrieves the speed of a vehicle when it passes a specific point (e.g., a speed camera).

    Args:
    vehicle_id (str): The ID of the vehicle.
    camera_x (float): The x-coordinate of the speed camera.
    camera_y (float): The y-coordinate of the speed camera.
    tolerance (float): The distance tolerance to consider the vehicle at the camera point.

    Returns:
    float: The speed of the vehicle at the camera point in m/s, or None if the vehicle is not at the point.
    """
    try:
        # Get the current position of the vehicle
        x, y = traci.vehicle.getPosition(vehicle_id)

        # Calculate the distance from the vehicle to the camera
        distance = ((x - radar_x) ** 2 + (y - radar_y) ** 2) ** 0.5

        # Check if the vehicle is within the tolerance distance of the camera
        if distance <= tolerance:
            # Get the speed of the vehicle
            speed = traci.vehicle.getSpeed(vehicle_id)
            return speed
        else:
            return None

    except Exception as e:
        print(f"Error retrieving speed: {e}")
        return None

class IncidentType(Enum):
    unexpected_brake = 1
    unexpected_lane_change = 2
    aggressive_lane_change = 3
    reduce_headway = 4

def provoke_incident(vehicle_id, incident_type):
    incidentProvoked = False

    if incident_type == IncidentType.unexpected_brake:
        # Unexpected brake
        traci.vehicle.setSpeed(vehicle_id, np.random.randint(0, 15))
        incidentProvoked = True
    elif incident_type == IncidentType.unexpected_lane_change:
        # Force an unsafe lane change
        traci.vehicle.changeLane(vehicle_id, 1, 10)  # Change to lane 1 immediately
        incidentProvoked = True
    elif incident_type == IncidentType.aggressive_lane_change:
        # Disable safety checks
        traci.vehicle.setSpeedMode(vehicle_id, 0)  # Disable all safety checks
        traci.vehicle.setLaneChangeMode(vehicle_id, 0)  # Allow aggressive lane changing
        incidentProvoked = True
    elif incident_type == IncidentType.reduce_headway:
        # Adjust car-following model parameters to reduce safety
        # traci.vehicle.setParameter(vehicle_id, "carFollowModel.tau", "0.5")  # Reduce headway time
        # incidentProvoked = True
        pass # FIXME: do nothing for now, it will be refined in the future
    else:
        return incidentProvoked