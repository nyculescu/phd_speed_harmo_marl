import numpy as np
import math
import os

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
        self._veh_depart = 0

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open(f"{os.getcwd()}\\marl\\configs\\sumo\\main_road_vehicles.rou.xml", "w") as routes:
            print("""<routes>
	<vType accel="1.0" decel="4.5" id="normal_car_s" length="5.0" minGap="2.5" maxSpeed="42" sigma="0.5" desiredMaxSpeed="36"/>
	<vType accel="1.5" decel="4.0" id="sporty_car_s" length="5.0" minGap="2.5" maxSpeed="67" sigma="0.6" desiredMaxSpeed="42"/>
	<vType accel="1.0" decel="4.5" id="trailer_s" length="10.0" minGap="2.5" maxSpeed="30" sigma="0.5" desiredMaxSpeed="30"/>
	<vType accel="1.0" decel="4.5" id="coach_s" length="15.0" minGap="2.5" maxSpeed="25" sigma="0.3" desiredMaxSpeed="25" />

    <route id="left_to_right" edges="LRE LRL LRS"/>
                  """, file=routes)

            """
            The following parameters affect the operation of the friction device.
                device.friction.stdDev: standard deviation when adding gaussian noise (default 1)
                device.friction.offset: constant offset to apply to all friction values (default 0)
            Reference: https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#devices


            """

            veh_hazard = np.random.randint(0, car_gen_steps)

            for veh_cnt, step in enumerate(car_gen_steps):
                veh_type_int = np.random.randint(0, 4)
                if (veh_type_int == 0):
                    veh_type = "normal_car_s"
                    veh_maxDepartSpeed = np.random.randint(25, 28)
                elif (veh_type_int == 1):
                    veh_type = "sporty_car_s"
                    veh_maxDepartSpeed = np.random.randint(28, 32)
                elif (veh_type_int == 2):
                    veh_type = "trailer_s"
                    veh_maxDepartSpeed = np.random.randint(20, 26)
                else:
                    veh_type = "coach_s"
                    veh_maxDepartSpeed = np.random.randint(20, 25)
                
                self._veh_depart = self._veh_depart + np.random.randint(0, 20.0)

                print(f"""\t<vehicle id="{veh_cnt}" \
type="{veh_type}" \
depart="{self._veh_depart}" \
from="R0.E" to="R0.S" departSpeed="random" maxDepartSpeed="{veh_maxDepartSpeed}" \
route="left_to_right" departLane="best">\
\n\t\t<param key="has.friction.device" value="true"/>\
\n\t</vehicle>""", file=routes)

            print("""
</routes>""", file=routes)
