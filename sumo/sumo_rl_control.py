import traci
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

sumoCmd = [
    sumoBinary,
    "-c",
    os.path.join(BASE_DIR, "cross.sumocfg")
]

traci.start(sumoCmd)

traffic_light_id =  "center" #traci.trafficlight.getIDList()[0]

step = 0

while step < 1000:
    traci.simulationStep()
    time.sleep(0.3)

    if step % 30 == 0:
        current_phase = traci.trafficlight.getPhase(traffic_light_id)
        next_phase = (current_phase + 1) % 4
        traci.trafficlight.setPhase(traffic_light_id, next_phase)

    step += 1

traci.close()