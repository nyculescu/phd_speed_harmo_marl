from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from models import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path
import winsound

if __name__ == "__main__":

    config = import_train_configuration(config_file=f'{os.getcwd()}\\marl\\configs\\training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    trafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    visualization = Visualization(
        path, 
        dpi=96
    )

    simulation = Simulation(
        model,
        memory,
        trafficGen,
        config['gamma'], # gamma param. of the Bellman equation
        config['max_steps'], # the duration of each episode. 1 step = 1 sec.
        sumo_cmd,
        config['num_states'], #  the size of the state of the env from the agent perspective (a change here also requires algorithm changes).
        config['num_actions'], # the number of possible actions (a change here also requires algorithm changes).
        config['training_epochs'] # the number of training iterations executed at the end of each episode.
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1
        winsound.Beep(440, 700)

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    model.save_model(path)

    copyfile(src=f'{os.getcwd()}\\marl\\configs\\training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    visualization.save_data_and_plot(data=simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    visualization.save_data_and_plot(data=simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    visualization.save_data_and_plot(data=simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')