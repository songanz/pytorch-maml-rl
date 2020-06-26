import numpy as np

from . import driveFuncs
from . import defineCnst as C
from . import params
from .Car import Car

""" THIS IS FOR ATTACK ONLY """

def initCarsAtt(numCar, numAtt):
    # no need to put the attacker near the target agent anymore
    envCars = initCars(numCar)
    envCars = insertion_sort(envCars, numAtt)

    return envCars

def insertion_sort(envCars, numAtt):
    for i in range(1, numAtt+1):
        minIndx = i
        dist2Target = envCars[i].xPos ** 2 + envCars[i].yPos ** 2
        for j in range(i, len(envCars)):
            curDis2Tar = envCars[j].xPos ** 2 + envCars[j].yPos ** 2
            if curDis2Tar < dist2Target:
                dist2Target = curDis2Tar
                minIndx = j
        swap2Cars(envCars, i, minIndx)

    return envCars

def swap2Cars(l, pos1, pos2):
    # pos1 id:
    id1 = l[pos1].id
    id2 = l[pos2].id

    # swap both the position in list and the id
    l[pos1], l[pos2] = l[pos2], l[pos1]
    l[pos1].id = id1
    l[pos2].id = id2

    return l

def initCars(numCar):
    # create new instance of car every episode
    envCars = []
    for nC in range(numCar):
        envCars.append(Car(nC))

    # place the target car (the training agent); C.T_CAR=0
    envCars[C.T_CAR].xPos = 0
    envCars[C.T_CAR].xVel = params.CAR_MIN_SPEED + np.random.randint(0, 6) * params.CAR_ACCEL_RATE
    envCars[C.T_CAR].laneNum = np.random.randint(3) * params.ROAD_CENTER_LANE  # any of the three lane
    envCars[C.T_CAR].yPos = envCars[0].laneNum * params.ROAD_LN_GAP + params.ROAD_LANE_CENTER

    # place rest of the cars
    driveFuncs.placeCars(envCars)

    return envCars

def getAffIndiOfTarget(car, envCars):
    actTar = envCars[C.T_CAR].ActNum
    xTar_rel = envCars[C.T_CAR].xPos - car.xPos
    vTar = envCars[C.T_CAR].xVel  # actual velocity
    yTar = envCars[C.T_CAR].laneNum  # actual y position

    afforIndOfTar = np.array([car.laneNum, car.xVel,
                              actTar, xTar_rel, vTar, yTar])

    affIndiOfaCar, _ = car.getAffIndiOfaCar(envCars)  # get affordance of a car
    afforIndOfTar = np.hstack((afforIndOfTar, affIndiOfaCar[2:]))

    return afforIndOfTar  # np.array (24,)

def scaleStateAtt(afIndNum):
    sScl = np.zeros(params.CAR_NUM_STATES_ATT)

    carIdx = np.arange(1, 8)  # car number 1 to 7
    xIdx = carIdx * 3
    vIdx = carIdx * 3 + 1
    yIdx = carIdx * 3 + 2

    # car
    sScl[0] = afIndNum[0] / params.ROAD_LEFT_LANE
    sScl[1] = (afIndNum[1] - params.CAR_MIN_SPEED) / params.CAR_SPD_RANGE
    sScl[2] = afIndNum[2] / params.CAR_NUM_ACTIONS
    # envi
    sScl[xIdx] = afIndNum[xIdx] / params.CAR_MAXVIS_DISTANCE  # relative x position
    sScl[vIdx] = (afIndNum[vIdx] - afIndNum[1]) / params.CAR_SPD_RANGE  # relative velocity
    sScl[yIdx] = (afIndNum[yIdx] - afIndNum[0]) / params.ROAD_LEFT_LANE  # relative y position

    return np.around(sScl, decimals=4)

def checkCollisionID(car, envCars):
    """
    check collision and return both collision and car id
    :param car: Class Car, of the checking car
    :param envCars: list of Class Car, all the cars in the environment
    :return: collision: boolean; idCar: int, id of the car
    """
    collision = False
    idCar = -1
    for i in range(len(envCars)):
        if envCars[i] == car:
            continue
        xCar = envCars[i].xPos
        yCar = envCars[i].yPos

        if abs(xCar - car.xPos) < C.X_COL_LMT and abs(yCar - car.yPos) < C.Y_COL_LMT:
            collision = True
            idCar = i
            break

    return collision, idCar

def whichLnMarker(lnNum):

    if abs(lnNum - 10) < 2:
        return 10
    if abs(lnNum - 6) < 2:
        return 6
    if abs(lnNum - 2) < 2:
        return 2
    if abs(lnNum + 2) < 2:
        return -2

def rewRespSensAtt(envCars, collisionTar, collisionAtt, id4OtherColliCar):
    """
    failureCode:
    -1: no collision of the target, but the attacker may collide to other env. cars
    0: the other car's fault

    1: no car is changing lane; target car is crashed from behind, when braking
    2: no car is changing lane; target car crash to the front car, with evasive action
    3: no car is changing lane; target car crash to the front car, without evasive action

    4: target car is changing lane; target car crash to the tageted lane env. car

    5: both car change lane and the target car is the left car who is response to the crash
    6: both car change lane and the target car is the left car who is response to the crash, with evasive action

    7: both car change lane and the env car is the left car who is response to the crash, with evasive action
    """
    # get affordance, not scaled yet
    idC_N, indx = driveFuncs.getAffordInd(envCars)  # get next state for all cars
    rew = -0.05
    failureCode = -1  # if no failure

    """ target not in the near vehicles of the attacker """
    if C.T_CAR not in indx:
        rew = -1
        return rew, failureCode

    """ target no collision but attacker collide to env cars """
    if not collisionTar and collisionAtt:
        rew = -1
        return rew, failureCode

    """ No collision """
    if not collisionTar:
        return rew, failureCode

    """ Collision and attacker is near the target """
    actTar = envCars[C.T_CAR].ActNum
    actColli = envCars[id4OtherColliCar].ActNum

    xTar = envCars[C.T_CAR].xPos
    lnTar = envCars[C.T_CAR].laneNum
    vTar = envCars[C.T_CAR].xVel

    xColli = envCars[id4OtherColliCar].xPos
    lnColli = envCars[id4OtherColliCar].laneNum
    vColli = envCars[id4OtherColliCar].xVel

    """ collision between target and id4OtherColliCar """
    if not driveFuncs.isOnLaneMark(lnTar) and not driveFuncs.isOnLaneMark(lnColli):
        """ No car is changing lane """
        if xTar > xColli:
            """ Target AV in front """
            if actColli == 2 or actColli == 3 or actColli == 6 or actColli == 7:
                # following car did evasive action
                rew = -0.5
                failureCode = 1
                return rew, failureCode
            else:
                # following car no evasive action
                rew = -1
                failureCode = 0
                return rew, failureCode
        else:
            """ Targe AV is behind """
            if actTar == 2 or actTar == 3 or actTar == 6 or actTar == 7:
                # following car did evasive action
                rew = 0.5
                failureCode = 2
                return rew, failureCode
            else:
                # following car no evasive action
                rew = 1
                failureCode = 3
                return rew, failureCode

    elif driveFuncs.isOnLaneMark(lnTar) and (not driveFuncs.isOnLaneMark(lnColli)):
        """ Target AV is changing lane """
        lnTar_N = lnTar + envCars[C.T_CAR].laneNumAct
        if (lnColli > lnTar and lnTar_N >= lnTar) or (lnColli < lnTar and lnTar_N <= lnTar):
            # agent is on the target car's targeted lane
            rew = 1
            failureCode = 4  # target car change vehicle wrongly
            return rew, failureCode
        else:
            # agent is on the target car's original lane
            if xTar > xColli:
                """ Target AV in front """
                if actColli == 2 or actColli == 3 or actColli == 6 or actColli == 7:
                    # following car did evasive action
                    rew = -0.5
                    failureCode = 1
                    return rew, failureCode
                else:
                    # following car no evasive action
                    rew = -1
                    failureCode = 0
                    return rew, failureCode
            else:
                """ Targe AV is behind """
                if actTar == 2 or actTar == 3 or actTar == 6 or actTar == 7:
                    # following car did evasive action
                    rew = 0.5
                    failureCode = 2
                    return rew, failureCode
                else:
                    # following car no evasive action
                    rew = 1
                    failureCode = 3
                    return rew, failureCode

    elif (not driveFuncs.isOnLaneMark(lnTar)) and driveFuncs.isOnLaneMark(lnColli):
        """ The other car is changing lane """
        lnColli_N = lnColli + envCars[id4OtherColliCar].laneNumAct
        if (lnTar > lnColli and lnColli_N >= lnColli) or (lnTar < lnColli and lnColli_N <= lnColli):
            # target AV is on the other car targeted lane
            rew = -1
            failureCode = 0
            return rew, failureCode
        else:
            # target AV is on the agent car's original lane
            if xTar > xColli:
                """ Target AV in front """
                if actColli == 2 or actColli == 3 or actColli == 6 or actColli == 7:
                    # following car did evasive action
                    rew = -0.5
                    failureCode = 1
                    return rew, failureCode
                else:
                    # following car no evasive action
                    rew = -1
                    failureCode = 0
                    return rew, failureCode
            else:
                """ Targe AV is behind """
                if actTar == 2 or actTar == 3 or actTar == 6 or actTar == 7:
                    # following car did evasive action
                    rew = 0.5
                    failureCode = 2
                    return rew, failureCode
                else:
                    # following car no evasive action
                    rew = 1
                    failureCode = 3
                    return rew, failureCode
    else:
        """ Two cars are changing lane """
        if whichLnMarker(lnTar) == whichLnMarker(lnColli):
            if xTar > xColli:
                """ Target AV in front """
                if actColli == 2 or actColli == 3 or actColli == 6 or actColli == 7:
                    # following car did evasive action
                    rew = -0.5
                    failureCode = 1
                    return rew, failureCode
                else:
                    # following no did evasive action
                    rew = -1
                    failureCode = 0
                    return rew, failureCode
            else:
                """ Targe AV is behind """
                if actTar == 2 or actTar == 3 or actTar == 6 or actTar == 7:
                    # following car did evasive action
                    rew = 0.5
                    failureCode = 2
                    return rew, failureCode
                else:
                    # following car no evasive action
                    rew = 1
                    failureCode = 3
                    return rew, failureCode

        elif whichLnMarker(lnTar) > whichLnMarker(lnColli):
            """ 
            change lane to the middle lane from left and right lane
            right lane car get the right of road 
            """
            # Target car is the left car: responsible to the crash
            if actTar == 4 or actTar == 6:
                # with evasive action
                rew = 0.5
                failureCode = 6
                return rew, failureCode
            # without evasive action
            rew = 1
            failureCode = 5
            return rew, failureCode
        else:
            """ 
            change lane to the middle lane from left and right lane
            right lane car get the right of road 
            """
            # the other car is the left car: responsible to the crash
            if actColli == 4 or actColli == 6:
                # with evasive action
                rew = -0.5
                failureCode = 7
                return rew, failureCode
            # without evasive action
            rew = -1
            failureCode = 0
            return rew, failureCode

def rewRespSens(envCars, collisionEgo, id4OtherColliCar):
    """
    For evaluation without attacker

    failureCode:
    -1: no collision of the target, but the attacker may collide to other env. cars
    0: the other car's fault

    1: no car is changing lane; target car is crashed from behind, when braking
    2: no car is changing lane; target car crash to the front car, with evasive action
    3: no car is changing lane; target car crash to the front car, without evasive action

    4: target car is changing lane; target car crash to the tageted lane env. car

    5: both car change lane and the target car is the left car who is response to the crash
    6: both car change lane and the target car is the left car who is response to the crash, with evasive action

    7: both car change lane and the env car is the left car who is response to the crash, with evasive action
    """

    failureCode = -1  # if no failure

    if not collisionEgo:
        return failureCode

    """ Collision and attacker is near the target """
    actEgo = envCars[C.T_CAR].ActNum
    actColli = envCars[id4OtherColliCar].ActNum

    xEgo = envCars[C.T_CAR].xPos
    lnEgo = envCars[C.T_CAR].laneNum
    vTar = envCars[C.T_CAR].xVel

    xColli = envCars[id4OtherColliCar].xPos
    lnColli = envCars[id4OtherColliCar].laneNum
    vColli = envCars[id4OtherColliCar].xVel

    """ collision between target and id4OtherColliCar """
    if not driveFuncs.isOnLaneMark(lnEgo) and not driveFuncs.isOnLaneMark(lnColli):
        """ No car is changing lane """
        if xEgo > xColli:
            """ Target AV in front """
            if actColli == 2 or actColli == 3 or actColli == 6 or actColli == 7:
                # following car did evasive action
                failureCode = 1
                return failureCode
            else:
                # following car no evasive action
                failureCode = 0
                return failureCode
        else:
            """ Targe AV is behind """
            if actEgo == 2 or actEgo == 3 or actEgo == 6 or actEgo == 7:
                # following car did evasive action
                failureCode = 2
                return failureCode
            else:
                # following car no evasive action
                failureCode = 3
                return failureCode

    elif driveFuncs.isOnLaneMark(lnEgo) and (not driveFuncs.isOnLaneMark(lnColli)):
        """ Target AV is changing lane """
        lnEgo_N = lnEgo + envCars[C.T_CAR].laneNumAct
        if (lnColli > lnEgo and lnEgo_N >= lnEgo) or (lnColli < lnEgo and lnEgo_N <= lnEgo):
            # agent is on the target car's targeted lane
            failureCode = 4  # target car change vehicle wrongly
            return failureCode
        else:
            # agent is on the target car's original lane
            if xEgo > xColli:
                """ Target AV in front """
                if actColli == 2 or actColli == 3 or actColli == 6 or actColli == 7:
                    # following car did evasive action
                    failureCode = 1
                    return failureCode
                else:
                    # following car no evasive action
                    failureCode = 0
                    return failureCode
            else:
                """ Targe AV is behind """
                if actEgo == 2 or actEgo == 3 or actEgo == 6 or actEgo == 7:
                    # following car did evasive action
                    failureCode = 2
                    return failureCode
                else:
                    # following car no evasive action
                    failureCode = 3
                    return failureCode

    elif (not driveFuncs.isOnLaneMark(lnEgo)) and driveFuncs.isOnLaneMark(lnColli):
        """ The other car is changing lane """
        lnColli_N = lnColli + envCars[id4OtherColliCar].laneNumAct
        if (lnEgo > lnColli and lnColli_N >= lnColli) or (lnEgo < lnColli and lnColli_N <= lnColli):
            # target AV is on the other car targeted lane
            failureCode = 0
            return failureCode
        else:
            # target AV is on the agent car's original lane
            if xEgo > xColli:
                """ Target AV in front """
                if actColli == 2 or actColli == 3 or actColli == 6 or actColli == 7:
                    # following car did evasive action
                    failureCode = 1
                    return failureCode
                else:
                    # following car no evasive action
                    failureCode = 0
                    return failureCode
            else:
                """ Targe AV is behind """
                if actEgo == 2 or actEgo == 3 or actEgo == 6 or actEgo == 7:
                    # following car did evasive action
                    failureCode = 2
                    return failureCode
                else:
                    # following car no evasive action
                    failureCode = 3
                    return failureCode
    else:
        """ Two cars are changing lane """
        if whichLnMarker(lnEgo) == whichLnMarker(lnColli):
            if xEgo > xColli:
                """ Target AV in front """
                if actColli == 2 or actColli == 3 or actColli == 6 or actColli == 7:
                    # following car did evasive action
                    failureCode = 1
                    return failureCode
                else:
                    # following no did evasive action
                    failureCode = 0
                    return failureCode
            else:
                """ Targe AV is behind """
                if actEgo == 2 or actEgo == 3 or actEgo == 6 or actEgo == 7:
                    # following car did evasive action
                    failureCode = 2
                    return failureCode
                else:
                    # following car no evasive action
                    failureCode = 3
                    return failureCode

        elif whichLnMarker(lnEgo) > whichLnMarker(lnColli):
            """ 
            change lane to the middle lane from left and right lane
            right lane car get the right of road 
            """
            # Target car is the left car: responsible to the crash
            if actEgo == 4 or actEgo == 6:
                # with evasive action
                failureCode = 6
                return failureCode
            # without evasive action
            failureCode = 5
            return failureCode
        else:
            """ 
            change lane to the middle lane from left and right lane
            right lane car get the right of road 
            """
            # the other car is the left car: responsible to the crash
            if actColli == 4 or actColli == 6:
                # with evasive action
                failureCode = 7
                return failureCode
            # without evasive action
            failureCode = 0
            return failureCode
