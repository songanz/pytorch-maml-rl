import numpy as np

from . import driveFuncs
from . import defineCnst as C
from . import params
from . import utils_sim
import torch as tr
from torch.autograd import Variable
import scipy.io as sio
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

""" THIS IS FOR ATTACK ONLY """

def initCarsAtt(numCar, num_attacker = 1):

    idxs4Attacker = -1*np.ones((6, ))
    while np.all(idxs4Attacker[3:] == -1):
        envCars = utils_sim.initCars(numCar)

        # get the near cars index of the attacker
        # and sway the id of one of near cars to 1 (target)
        _, idxs4Attacker = driveFuncs.getAffordInd(envCars)

    if C.T_CAR in idxs4Attacker[3:]:
        return envCars

    # randomly choose a near car as the target and make sure the target is at position 1 with id 1
    pos4NearCar = -1
    while pos4NearCar == -1:
        pos4NearCar = random.sample(list(idxs4Attacker[3:]), 1)[0]
    swap2Cars(envCars, pos4NearCar, C.T_CAR)

    return envCars

def debugVis(envCars, visId):
    plt.cla()  # clear current axis
    ax = plt.gca()
    ax.set_xlim(-100, 100)
    ax.set_ylim(-0.1, 10.9)
    ax.set_aspect('equal')
    plt.plot([-100, 100], [0, 0], "-k")  # black: lane boundary
    plt.plot([-100, 100], [3.6, 3.6], "-k")
    plt.plot([-100, 100], [7.2, 7.2], "-k")
    plt.plot([-100, 100], [10.8, 10.8], "-k")

    for ind in range(len(envCars)):

        car = mpatches.Rectangle((envCars[ind].xPos - 2.2, envCars[ind].yPos - 1.1), 4.4, 2.2)

        if ind == C.T_CAR:
            # target is in blue
            car.set_fill(True)
            car.set_color("blue")
        elif ind == C.E_CAR:
            # attacker is in red
            car.set_fill(True)
            car.set_color("red")
        elif ind in visId:
            # other cars in near set to green
            car.set_fill(True)
            car.set_color("green")
        else:
            car.set_fill(False)

        ax.add_patch(car)

    plt.grid(False)
    plt.pause(0.001)

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

def swap2Cars(l, pos1, pos2):
    # pos1 id:
    id1 = l[pos1].id
    id2 = l[pos2].id

    # swap both the position in list and the id
    l[pos1], l[pos2] = l[pos2], l[pos1]
    l[pos1].id = id1
    l[pos2].id = id2

    return l

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
    actEgo = envCars[C.E_CAR].ActNum
    actColli = envCars[id4OtherColliCar].ActNum

    xEgo = envCars[C.E_CAR].xPos
    lnEgo = envCars[C.E_CAR].laneNum
    vTar = envCars[C.E_CAR].xVel

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
        lnEgo_N = lnEgo + envCars[C.E_CAR].laneNumAct
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

def drawFrame(postAtTime, visId):
    ax = plt.gca()
    ax.set_xlim(-100, 100)
    ax.set_ylim(-0.1, 10.9)
    ax.set_aspect('equal')
    plt.plot([-100, 100], [0, 0], "-k")  # black: lane boundary
    plt.plot([-100, 100], [3.6, 3.6], "-k")
    plt.plot([-100, 100], [7.2, 7.2], "-k")
    plt.plot([-100, 100], [10.8, 10.8], "-k")

    for ind in range(len(postAtTime)):

        car = mpatches.Rectangle((postAtTime[ind, 0] - 2.2, postAtTime[ind, 1] - 1.1), 4.4, 2.2)

        if ind == C.T_CAR:
            # target is in blue
            car.set_fill(True)
            car.set_color("blue")
        elif ind == C.E_CAR:
            # attacker is in red
            car.set_fill(True)
            car.set_color("red")
        elif ind in visId:
            # other cars in near set to green
            car.set_fill(True)
            car.set_color("green")
        else:
            car.set_fill(False)

        ax.add_patch(car)

    plt.draw()


def videoGenerate(Code, Count, postHist, visIdHist, totalIdx, numCar):
    """
    Save video
    posHist[i][timeIdx] = np.array([envCars[i].xPos, envCars[i].yPos, envCars[i].xVel, envCars[i].xVelAct])
    """
    vidoedir = os.getcwd() + "/out/videos/" + "Failure" + str(int(Code)) + "/"
    if not os.path.exists(vidoedir):
        os.makedirs(vidoedir)
    videoFile = vidoedir + "numCar" + str(int(numCar)) + "Failure" + str(int(Code)) + "_Num" + str(int(Count)) + ".mp4"
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=1/params.SIM_TIME_STEP*10)  # for accelerate the animation, times 10
    fig = plt.figure(figsize=(20,2))

    with writer.saving(fig, videoFile, 100):
        for t in range(totalIdx+1):
            plt.cla()
            drawFrame(postHist[:,t,:], visIdHist[t])
            writer.grab_frame()


def evalEpisodeAtt(attacker_agent, target_agent, envCars, i, numColi, device, saveHist=False, debug=False,
                   failureCodeCount=np.zeros((params.FAILURE_CODE_NUM, )),
                   attColiCount=np.zeros((params.FAILURE_CODE_NUM, )), saveVideo=False):
    """

    :param attacker_agent:
    :param target_agent:
    :param envCars: list of Class Car, all cars in the environment
    :param i: ith episode
    :param numColi:
    :param device:
    :param saveHist:
    :param debug:
    :param failureCodeCount:
    :param attColiCount: whether collide with the attacker
    :param saveVideo:
    :return:
    """
    numCar = len(envCars)

    # History for animation in matlab
    posHist, visIdHist, actChoice, appliedAction, rewHist = utils_sim.initAniHist(numCar)

    # get initial position store it for animation
    idC, visIdHist[0] = driveFuncs.getAffordInd(envCars)
    attackerState = getAffIndiOfTarget(envCars[0], envCars)  # np.array (24,)

    """ FOR DEBUG """
    if debug and not saveVideo:
        debugVis(envCars, visIdHist[0])

    # Store initial position
    driveFuncs.storeCarsPos(envCars, posHist, 0)
    s0Attacker = scaleStateAtt(attackerState)  # get the current scaled state for ego car
    s0Target = driveFuncs.scaleState(idC[C.T_CAR, :])  # get the current scaled state for target car

    # Result variables
    totRew = 0  # total reward per episode

    s0MatPE = []
    qEstMatPE = []
    actAndAplAct = []
    stateidC = []
    allRew = []
    rewParam = []
    # velHist = deque(maxlen=3)

    prfActTar = 0

    for j in range(params.NUM_STEP):
        # eps = np.random.random()
        attacker_agent.Q_eval_net.eval()
        Q, _, _ = attacker_agent.Q_eval_net(
            Variable(tr.from_numpy(s0Attacker).to(device), requires_grad=False).float()[None, ...])
        actAtt = int(tr.argmax(Q))

        # target agent
        QTar, _, _ = target_agent.Q_eval_net(
            Variable(tr.from_numpy(s0Target).to(device), requires_grad=False).float()[None, ...])
        actTar = int(tr.argmax(QTar))

        """ safe act policy for target car """
        e_ln = idC[C.T_CAR, C.E_LN]
        if driveFuncs.isOnLaneMark(e_ln):
            # On lane mark
            if prfActTar != 0:
                # Override learned act if we have preferred action for lane changes
                actTar = prfActTar
        if QTar.cpu().data.numpy()[0, 6] > QTar.cpu().data.numpy()[0, 7]:
            sfLMAct = 6
        else:
            sfLMAct = 7

        # Check safety controller, override to new action if needed
        newAct, rstPrfAct = driveFuncs.safeActEval(actTar, idC[C.T_CAR, :], sfLMAct)

        if newAct != actTar:
            # saftey controller overrides the first choice, set the original action for collision
            # print('safe action')
            actTar = newAct

        """ constraint on the action of the attacker for more efficient attacking """
        lnAtt = idC[C.E_CAR, C.E_LN]
        if lnAtt == params.ROAD_RIGHT_LANE:
            # on right lane
            if actAtt == C.A_BR or actAtt == C.A_MR:
                actAtt = C.A_MM

        elif lnAtt == params.ROAD_LEFT_LANE:
            # on left lane
            if actAtt == C.A_BL or actAtt == C.A_ML:
                actAtt = C.A_MM

        # store applied action for the attacker car
        appliedAction[j] = actAtt

        """ Update cars with new actions """
        driveFuncs.setEgoCarAction(envCars[C.E_CAR], actAtt)  # set action for the ego car
        driveFuncs.setEgoCarAction(envCars[C.T_CAR], actTar)  # set action for the target car
        driveFuncs.setAction(envCars[2:], idC[2:, :])  # set actions for env cars
        driveFuncs.appAction(envCars)  # Apply actions

        idC_N, visIdHist[j + 1] = driveFuncs.getAffordInd(envCars)  # get next state for all cars
        attackerState_N = getAffIndiOfTarget(envCars[0], envCars)  # np.array (24,)

        driveFuncs.storeCarsPos(envCars, posHist, j + 1)  # Store position

        """ FOR DEBUG """
        if debug and not saveVideo:
            debugVis(envCars, visIdHist[j+1])

        """ 
        Check collision of the target car
        Target car not neccesarily collide into the attacker 
        """
        collisionTar, id4OtherColliCar = checkCollisionID(envCars[C.T_CAR], envCars)
        collisionAtt, _ = checkCollisionID(envCars[C.E_CAR], envCars)

        # Get new scaled attacker car state, and target car state
        s1Attacker = scaleStateAtt(attackerState_N)
        s1Target = driveFuncs.scaleState(idC_N[C.T_CAR, :])

        """ write reward code here """
        rew, failureCode = rewRespSensAtt(envCars, collisionTar, collisionAtt, id4OtherColliCar)

        if collisionTar:
            numColi += 1
            failureCodeCount[failureCode] += 1
            if id4OtherColliCar == C.E_CAR:
                attColiCount[failureCode] += 1
            if saveVideo and failureCode != 0:  # failureCode=0: not responsible case, dont save that
                videoGenerate(failureCode, failureCodeCount[failureCode], posHist, visIdHist, j+1, numCar)

        totRew += rew  # total reward for implemented actions
        rewHist[j] = rew

        """ save prfActTar for target vehicle's safety check WORK ON THIS CODE """
        if idC[C.T_CAR, C.E_LN] % 4 == 0 and actTar > C.A_HM:
            # ego car: previously on lane, applied action was either move left or move right
            # save it for future
            prfActTar = actTar
        if rstPrfAct or idC_N[C.E_CAR, C.E_LN] % 4 == 0:
            # reset as there is no preferred action once we reach next lane
            prfActTar = 0

        if collisionTar or collisionAtt or (C.T_CAR not in visIdHist[j + 1]):
            break
        else:
            # no collision so continue
            idC, s0Attacker, s0Target = idC_N, s1Attacker, s1Target

    if saveHist:
        fileName = os.path.abspath('out/animation_zsa/data/AttposHistEval')
        sio.savemat(fileName,
                    {'posHist': posHist,
                     'numCar': numCar,
                     'visIdHist': visIdHist,
                     'appliedAction': appliedAction,
                     'actChoice': actChoice,
                     'k': j + 1,
                     's0Mat': s0MatPE,
                     'qEstMat': qEstMatPE,
                     'ep': i + 1,
                     'actAndAplAct': actAndAplAct,
                     'stateidC': stateidC,
                     'rewHist': rewHist,
                     'allRew': allRew,
                     'rewParam': rewParam})

    # All done result saved

    return rewHist, numColi, failureCodeCount, attColiCount

def evalPolicyAtt(attacker_agent, target_agent, device, ep, debug=False, saveVideo=False, saveHist=False):
    for numCar in range(10, 20):
        print(numCar)
        numColi = 0  # increment on collision
        failureCodeCount = np.zeros((params.FAILURE_CODE_NUM,))
        attColiCount = np.zeros((params.FAILURE_CODE_NUM,))

        for i in range(ep):
            # Initialize all car positions
            envCars = utils_sim.initCars(numCar)

            rewHist, numColi, failureCodeCount, attColiCount = \
                evalEpisodeAtt(attacker_agent, target_agent, envCars, i, numColi, device, saveHist=False, debug=debug,
                               failureCodeCount=failureCodeCount, attColiCount=attColiCount, saveVideo=saveVideo)

            if len(rewHist) > 100:
                meanRew = np.mean(rewHist[-100:])
            else:
                meanRew = np.mean(rewHist)

            print('episode:', i,
                  # ' num RL ', numRLAct, ' rd ', numRandAct, ' sf ', numSafeAct, ' tot ', j + 1,
                  # ' rew {:.2f}'.format(totRew),
                  'mean rew {:.2f}'.format(meanRew),
                  'num coli', numColi)

        # everything is done save result
        if saveHist:
            evalName = os.path.abspath('out/animation_zsa/AttEvalCarFrAccl2Pol%d.mat' % numCar)
            sio.savemat(evalName, {'rewHist': rewHist,
                                   'meanRewHist': meanRew,
                                   'numColi': numColi,
                                   'failureCodeCount': failureCodeCount,
                                   'attColiCount': attColiCount})

        print('num Col: ', numColi, ' for car: ', numCar)