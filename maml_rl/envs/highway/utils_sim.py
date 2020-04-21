import numpy as np
from . import params
from . import driveFuncs
from .Car import Car
from . import defineCnst as C


def initAniHist(numCar):
    posHist = np.zeros([numCar, params.NUM_STEP + 1, 4], dtype=float)
    visIdHist = np.ones([params.NUM_STEP + 1, 6],
                        dtype=np.int8) * -1  # there can be 5 visible cars around ego car to make things nicer
    actChoice = np.ones([params.NUM_STEP],
                        dtype=np.int8) * -1  # an array to indicate actions taken by 0 -- Random, 1 -- RL 2 -- Safety
    appliedAction = np.ones([params.NUM_STEP], dtype=np.int8) * -1  # an array to indicate actions taken 0 to 6
    rewHist = np.ones([params.NUM_STEP]) * -10
    return posHist, visIdHist, actChoice, appliedAction, rewHist


def initCars(numCar):
    # create new instance of car every episode
    envCars = []
    for nC in range(numCar):
        envCars.append(Car(nC))

    # place ego car; C.E_CAR=0
    envCars[C.E_CAR].xPos = 0
    envCars[C.E_CAR].xVel = params.CAR_MIN_SPEED + np.random.randint(0, 6) * params.CAR_ACCEL_RATE
    envCars[C.E_CAR].laneNum = np.random.randint(3) * params.ROAD_CENTER_LANE  # any of the three lane
    envCars[C.E_CAR].yPos = envCars[0].laneNum * params.ROAD_LN_GAP + params.ROAD_LANE_CENTER

    # place rest of the cars
    driveFuncs.placeCars(envCars)

    return envCars
