from . import params
from . import defineCnst as C
import numpy as np
import torch as tr


def placeCars(envCars, traffic_density):
    # do this for all the cars
    # index starts from 1 as ego car is car_0
    if len(envCars) > 15:
        MAX_DIST = params.SIM_MAX_DISTANCE*2
    else:
        MAX_DIST = params.SIM_MAX_DISTANCE  # params.SIM_MAX_DISTANCE = 240

    MAX_DIST = MAX_DIST * traffic_density
    SAFE_DISTANCE = params.CAR_SAFE_DISTANCE * traffic_density

    for i in range(1, len(envCars)):
        laneOverlap = True
        while laneOverlap:
            xPos = MAX_DIST * (np.random.random() - 0.5)  # place cars between -0.5*maxSimDist to +0.5*maxSimDist
            # random number of cars per lane 3 lanes shifted by 2 due to lane marking
            laneNum = np.random.randint(3)*params.ROAD_CENTER_LANE
            laneOverlap = False
            for j in range(0, i):
                # only 0 to i cars have been placed already. 0-->ego car
                if (laneNum == envCars[j].laneNum) and (abs(xPos-envCars[j].xPos) < SAFE_DISTANCE):
                    # at least 20m of gap is between two cars between center of cars it is 18m
                    laneOverlap = True
                    break   # overlap so start a new
            if not laneOverlap:
                # no break happened so assign x,y position and lanenumber to car i
                envCars[i].xPos = xPos
                # as of now use between min and max velocity must be changed latter
                envCars[i].xVel = params.CAR_MIN_SPEED + np.random.randint(0, 6) * params.CAR_ACCEL_RATE
                # 3.6 is the lane width and 1.8 is the center of the lane
                envCars[i].yPos = laneNum * params.ROAD_LN_GAP + params.ROAD_LANE_CENTER
                envCars[i].MaxVel = envCars[i].xVel
                envCars[i].laneNum = laneNum


def getAffordInd(envCars):
    # get affordance indicator
    # print(len(envCars))
    indiNum = np.ones([len(envCars), params.CAR_NUM_STATES], dtype='f')*-1e3
    tmp = np.ones([len(envCars), params.NUM_VIS_CARS], dtype=int)  # there are 6 vehicles which are visible to every car
    for i in range(len(envCars)):
        indiNum[i], tmp[i] = envCars[i].getAffIndiOfaCar(envCars)
    return indiNum, tmp[0]  # only the ego car used to indicate which cars are visible


def setAction(envCars, afInd):
    # set action for rest of the cars car
    # this only works for level 0
    # modify this to include level 1 policy for ego car later
    # as of now no acceleration and no lane change same as uofm
    for i in range(len(envCars)):
        # include based on front center car + accl if there is no other car
        # get the front car velocity and distance
        # IDM + accl

        cf_d = params.CAR_MAXVIS_DISTANCE
        cf_v = params.CAR_MAX_SPEED
        eg_v = afInd[i, C.E_SP]
        eg_ln = afInd[i, C.E_LN]
        eg_pos = envCars[i].xPos
        if eg_ln == params.ROAD_LEFT_LANE:
            # on left lane
            cf_d = afInd[i, C.FL_D]
            cf_v = afInd[i, C.FL_V]
        elif eg_ln == params.ROAD_CENTER_LANE:
            # on center lane
            cf_d = afInd[i, C.FC_D]
            cf_v = afInd[i, C.FC_V]
        elif eg_ln == params.ROAD_RIGHT_LANE:
            # on right lane
            cf_d = afInd[i, C.FR_D]
            cf_v = afInd[i, C.FR_V]

        if cf_d <= params.CAR_DISTANCE_1 and (eg_v-cf_v) >= params.CAR_ACCEL_RATE:
            envCars[i].xVelAct = params.CAR_HARD_DECEL_RATE
            envCars[i].yPosAct = params.CAR_MAINTAIN
            envCars[i].laneNumAct = params.CAR_MAINTAIN
            envCars[i].ActNum = C.A_HM
        elif (cf_d <= params.CAR_DISTANCE_0 and eg_v > params.CAR_MIN_SPEED) or \
             (cf_d <= params.CAR_DISTANCE_2 and (eg_v - cf_v) >= params.CAR_ACCEL_RATE):
            envCars[i].xVelAct = params.CAR_DECEL_RATE
            envCars[i].yPosAct = params.CAR_MAINTAIN
            envCars[i].laneNumAct = params.CAR_MAINTAIN
            envCars[i].ActNum = C.A_BM
        elif eg_pos < 170 and cf_d >= params.CAR_DISTANCE_3 and (cf_v - eg_v) >= params.CAR_ACCEL_RATE:
            # front car is far away and also moving away
            # ego car has enough space in front of it
            envCars[i].xVelAct = params.CAR_ACCEL_RATE
            envCars[i].yPosAct = params.CAR_MAINTAIN
            envCars[i].laneNumAct = params.CAR_MAINTAIN
            envCars[i].ActNum = C.A_AM
        else:
            envCars[i].xVelAct = params.CAR_MAINTAIN
            envCars[i].yPosAct = params.CAR_MAINTAIN
            envCars[i].laneNumAct = params.CAR_MAINTAIN
            envCars[i].ActNum = C.A_MM


def setEgoCarAction(egoCar, act):
    # set action for ego car
    # default is maintain
    egoCar.ActNum = act
    if act == C.A_AM:
        # accelerate
        egoCar.xVelAct = params.CAR_ACCEL_RATE
        egoCar.yPosAct = params.CAR_MAINTAIN
        egoCar.laneNumAct = params.CAR_MAINTAIN
    elif act == C.A_BM:
        # Hard accelerate
        egoCar.xVelAct = params.CAR_DECEL_RATE
        egoCar.yPosAct = params.CAR_MAINTAIN
        egoCar.laneNumAct = params.CAR_MAINTAIN
    elif act == C.A_HM:
        # decelerate
        egoCar.xVelAct = params.CAR_HARD_DECEL_RATE
        egoCar.yPosAct = params.CAR_MAINTAIN
        egoCar.laneNumAct = params.CAR_MAINTAIN
    elif act == C.A_ML:
        # change lane to left
        egoCar.xVelAct = params.CAR_MAINTAIN
        egoCar.yPosAct = params.CAR_TO_LEFT
        egoCar.laneNumAct = params.CAR_TO_LEFT_LN
    elif act == C.A_MR:
        # change lane to right
        egoCar.xVelAzzct = params.CAR_MAINTAIN
        egoCar.yPosAct = params.CAR_TO_RIGHT
        egoCar.laneNumAct = params.CAR_TO_RIGH_LN
    elif act == C.A_BL:
        # change lane to right
        egoCar.xVelAct = params.CAR_DECEL_RATE
        egoCar.yPosAct = params.CAR_TO_LEFT
        egoCar.laneNumAct = params.CAR_TO_LEFT_LN
    elif act == C.A_BR:
        # change lane to right
        egoCar.xVelAct = params.CAR_DECEL_RATE
        egoCar.yPosAct = params.CAR_TO_RIGHT
        egoCar.laneNumAct = params.CAR_TO_RIGH_LN
    else:
        # either invalid action number or 0 so maintain
        egoCar.xVelAct = params.CAR_MAINTAIN
        egoCar.yPosAct = params.CAR_MAINTAIN
        egoCar.laneNumAct = params.CAR_MAINTAIN


def appAction(envCars):
    # Action has been set now apply for every car
    egoCarVel = envCars[0].xVel
    for i in range(len(envCars)):
        if i > 0:
            # ego car has adjusted its velocity take new velocity into consideration
            egoCarVel = envCars[0].xVel
        envCars[i].stateUpdate(egoCarVel)


def storeCarsPos(envCars, posHist, timeIdx):
    for i in range(len(envCars)):
        # currently store x and y position for all the cars in the history array
        posHist[i][timeIdx] = np.array([envCars[i].xPos, envCars[i].yPos, envCars[i].xVel, envCars[i].xVelAct])


def scaleState(afIndNum):
    sScl = np.zeros(params.CAR_NUM_STATES)

    carIdx = np.arange(1, 7)  # car number 1 to 6
    xIdx = carIdx*3-1
    vIdx = carIdx*3
    yIdx = carIdx*3+1

    # car
    sScl[0] = afIndNum[0] / params.ROAD_LEFT_LANE
    sScl[1] = (afIndNum[1]-params.CAR_MIN_SPEED)/params.CAR_SPD_RANGE
    # envi
    sScl[xIdx] = afIndNum[xIdx]/params.CAR_MAXVIS_DISTANCE  # relative x position
    sScl[vIdx] = (afIndNum[vIdx]-afIndNum[1])/params.CAR_SPD_RANGE  # relative velocity
    sScl[yIdx] = (afIndNum[yIdx]-afIndNum[0])/params.ROAD_LEFT_LANE  # relative y position

    return np.around(sScl, decimals=4)

# for CVAE, calculate whole batch
def scaleStateTensor(afIndNum, device):
    sScl = tr.zeros(afIndNum.shape).to(device)

    carIdx = np.arange(1, 7)  # car number 1 to 6
    xIdx = carIdx*3-1
    vIdx = carIdx*3
    yIdx = carIdx*3+1

    # car
    sScl[:, 0] = afIndNum[:, 0] / params.ROAD_LEFT_LANE
    sScl[:, 1] = (afIndNum[:, 1] - params.CAR_MIN_SPEED)/ params.CAR_SPD_RANGE
    # envi
    sScl[:, xIdx] = afIndNum[:, xIdx]/params.CAR_MAXVIS_DISTANCE
    sScl[:, vIdx] = (afIndNum[:, vIdx]-tr.transpose(afIndNum[:, 1].repeat(6,1), 0,1))/params.CAR_SPD_RANGE
    sScl[:, yIdx] = (afIndNum[:, yIdx]-tr.transpose(afIndNum[:, 0].repeat(6,1), 0,1))/params.ROAD_LEFT_LANE

    return sScl


def reverseScaleStateTensor(afIndNum, device):
    sScl = tr.zeros(afIndNum.shape).to(device)

    carIdx = np.arange(1, 7)  # car number 1 to 6
    xIdx = carIdx*3-1
    vIdx = carIdx*3
    yIdx = carIdx*3+1

    # car
    sScl[:, 0] = afIndNum[:, 0] * params.ROAD_LEFT_LANE
    sScl[:, 1] = afIndNum[:, 1] * params.CAR_SPD_RANGE + params.CAR_MIN_SPEED
    # envi
    sScl[:, xIdx] = afIndNum[:, xIdx]*params.CAR_MAXVIS_DISTANCE  # relative x position
    sScl[:, vIdx] = afIndNum[:, vIdx]*params.CAR_SPD_RANGE + tr.transpose(sScl[:,1].repeat(6,1), 0,1)  # absolute velocity
    sScl[:, yIdx] = afIndNum[:, yIdx]*params.ROAD_LEFT_LANE + tr.transpose(sScl[:,0].repeat(6,1), 0,1)  # absolute y position

    return sScl

def checkColi4Ego(eIdC, eIdC_N, act):
    e_ln = eIdC[C.E_LN]
    collision = False
    if e_ln == params.ROAD_RIGHT_LANE and (act == C.A_BR or act == C.A_MR):
        # this implies some problem with safety controller
        collision = True
    elif e_ln == params.ROAD_LEFT_LANE and (act == C.A_BL or act == C.A_ML):
        # this implies some problem with safety controller
        collision = True
    else:
        # only this condition is valid
        for i in range(C.NUM_CAR_P_LN):
            fxPos = C.NUM_AFID_E+i*C.NUM_AFID_R  # 2+i*3: FL_D=2, FC_D=5, FR_D=8
            rxPos = fxPos+C.AFID_XPOS_OFFSET  # fxPos + 9: RL_D=11, RC_D=14, RR_D=17
            fyPos = fxPos+C.X_Y_POS_OFFSET   # fxPos + 2: FL_P=4, FC_P=7, FR_P=10
            ryPos = rxPos+C.X_Y_POS_OFFSET   # rxPos + 2: RL_P=13, RC_P=16, RR_P=19
            if (abs(eIdC_N[fyPos]-eIdC_N[C.E_LN])*C.LN_LENGTH < C.Y_COL_LMT and abs(eIdC_N[fxPos]) < C.X_COL_LMT) \
                    or (abs(eIdC_N[ryPos] - eIdC_N[C.E_LN]) * C.LN_LENGTH < C.Y_COL_LMT
                        and abs(eIdC_N[rxPos]) < C.X_COL_LMT):
                collision = True
    return collision


def rplAct(eg_v, cf_d, cf_v):
    if cf_d <= params.CAR_DISTANCE_1 and (eg_v-cf_v) >= params.CAR_ACCEL_RATE:
        return C.A_HM
    elif (cf_d <= params.CAR_DISTANCE_0 and eg_v > params.CAR_MIN_SPEED) \
            or chkSafeDist(eg_v, cf_d, cf_v) <= 0 \
            or (cf_d < params.CAR_DISTANCE_2 and (eg_v - cf_v) >= params.CAR_ACCEL_RATE):
        return C.A_BM
    else:
        return C.A_MM


def getVY(eIDC_N, cf_d):
    # this code works only when other cars are not changing the lane
    maxV = params.CAR_MIN_SPEED
    disY = eIDC_N[C.E_LN]

    eV = eIDC_N[C.E_SP]
    eP = eIDC_N[C.E_LN]

    # get the distance&velocity to closest front car
    fl_d, fc_d, fr_d = eIDC_N[C.FL_D], eIDC_N[C.FC_D], eIDC_N[C.FR_D]
    fl_v, fc_v, fr_v = eIDC_N[C.FL_V], eIDC_N[C.FC_V], eIDC_N[C.FR_V]

    # get the distance&velocity to closest rear car
    rl_d, rc_d, rr_d = eIDC_N[C.RL_D], eIDC_N[C.RC_D], eIDC_N[C.RR_D]
    rl_v, rc_v, rr_v = eIDC_N[C.RL_V], eIDC_N[C.RC_V], eIDC_N[C.RR_V]

    if fr_v > maxV or (fr_v == maxV and eP == C.R_LN):
        maxV = fr_v
        disY = C.R_LN
    if fl_v > maxV or (fl_v == maxV and eP == C.L_LN):
        maxV = fl_v
        disY = C.L_LN
    if fc_v > maxV or (fc_v == maxV and eP == C.C_LN):
        # by changing it to >= preference can be given for center lane
        maxV = fc_v
        disY = C.C_LN
    if abs(eV - maxV) < 1e-2:
        # dont penalize if ego car is going at maximum possible velocity
        disY = eP
    if (eP % 4) == 0 and cf_d > C.M_ACL_DIST:
        maxV = params.CAR_MAX_SPEED
        disY = eP

    if (eP % 4) == 0:
        # on lane mark
        if disY < eP:
            if eP == C.L_LN and (chkSafeDist(eV, fl_d, fl_v) <= 0 or
                                 chkSafeDist(eV, fc_d, fc_v) <= 0 or chkRearSafeDist(eV, rc_d, rc_v) <= 0):
                # currently on left lane but desired lane is right of me
                # not possible to change lane as it will be overwritten by safety
                maxV = eV
                disY = eP
            if eP == C.C_LN and (chkSafeDist(eV, fc_d, fc_v) <= 0 or
                                 chkSafeDist(eV, fr_d, fr_v) <= 0 or chkRearSafeDist(eV, rr_d, rr_v) <= 0):
                # currently on center lane but desired lane is right of me
                # not possible to change lane as it will be overwritten by safety
                maxV = eV
                disY = eP
        if disY > eP:
            if eP == C.R_LN and (chkSafeDist(eV, fr_d, fr_v) <= 0 or
                                 chkSafeDist(eV, fc_d, fc_v) <= 0 or chkRearSafeDist(eV, rc_d, rc_v) <= 0):
                # currently on right lane but desired lane is left of me
                # not possible to change lane as it will be overwritten by safety
                maxV = eV
                disY = eP
            if eP == C.C_LN and (chkSafeDist(eV, fc_d, fc_v) <= 0 or
                                 chkSafeDist(eV, fl_d, fl_v) <= 0 or chkRearSafeDist(eV, rl_d, rl_v) <= 0):
                # currently on center lane but desired lane is left of me
                # not possible to change lane as it will be overwritten by safety
                maxV = eV
                disY = eP
    return maxV, disY


def getcfDV(eIDC_N):
    eP = eIDC_N[C.E_LN]

    # get the distance to closest front car
    fl_d, fc_d, fr_d = eIDC_N[C.FL_D], eIDC_N[C.FC_D], eIDC_N[C.FR_D]
    fl_v, fc_v, fr_v = eIDC_N[C.FL_V], eIDC_N[C.FC_V], eIDC_N[C.FR_V]

    cf_d = params.CAR_DISTANCE_0

    if eP % C.C_LN == 0:
        # on center of lane
        if eP == C.R_LN:
            cf_d = fr_d
            cf_v = fr_v
        elif eP == C.C_LN:
            cf_d = fc_d
            cf_v = fc_v
        elif eP == C.L_LN:
            cf_d = fl_d
            cf_v = fl_v
    else:
        # between lanes
        if eP < C.C_LN:
            if fc_d < fr_d:
                cf_d = fc_d
                cf_v = fc_v
            else:
                cf_d = fr_d
                cf_v = fr_v
        elif eP < C.L_LN:
            if fc_d < fl_d:
                cf_d = fc_d
                cf_v = fc_v
            else:
                cf_d = fl_d
                cf_v = fl_v
        else:
            cf_d = fl_d
            cf_v = fl_v
    return cf_d, cf_v


def isOnLaneMark(ln):
    return (ln % 4) != 0


def getRewCmp(eIDC_N):
    # this would only work for level 1 - level 0 scenario where level 0 is modified IDM with no lane change option

    eV = eIDC_N[C.E_SP]
    eP = eIDC_N[C.E_LN]

    # get front dist and rel vel
    cf_d, cf_v = getcfDV(eIDC_N)  # Check this plz

    cf_relv = eV-cf_v

    # get feasible V and Y
    maxV, disY = getVY(eIDC_N, cf_d)  # Check this plz

    # between lanes when ego car is at a lower speed than maximum speed dont penalize
    rewV = 0
    if (eP % 4) == 0 or eV > maxV:
        # on lane penalize if going over or under speed
        # if eV is higher than maxV then penalize (this will happen only between lanes)
        rewV = np.exp(-(eV-maxV)**2/(2*2*params.CAR_ACCEL_RATE**2))-1

    # dont penalize when there is no front car
    rewY = 0
    if cf_d < C.M_ACL_DIST:
        # when there is a front car and there is a potential for ego car to achieve higher velocity
        rewY = np.exp(-(eP-disY)**2/(2*params.CAR_ACCEL_RATE**2))-1

    sfDist = (C.NOM_DIST * C.LEN_SCL) + (eV - cf_v) * C.NO_COLI_TIME
    midDis = sfDist+(C.M_ACL_DIST-sfDist)/C.LEN_SCL

    rewX = 0
    if (eP % 4) == 0 and cf_relv > 0 and cf_d < midDis:
            rewX = np.exp(-(cf_d-midDis)**2/(2*C.NOM_DIST**2))-1

    return rewV, rewY, rewX, cf_d, maxV, disY, eV, cf_v, midDis


# modified safety check based on inputs by Shankar see his slides
def chkRearSafeDist(e_v, r_d, r_v):
    if e_v < r_v:
        return -((r_d + C.NOM_DIST * C.LEN_SCL) - (e_v - r_v)*C.VEL_SCL)
    else:
        return -(r_d + C.NOM_DIST * C.LEN_SCL)


def chkSafeDist(e_v, f_d, f_v):
    # f_d,e_v,f_v all are +ve numbers
    if e_v > f_v:
        return (f_d - C.NOM_DIST * C.LEN_SCL) - (e_v - f_v) * C.NO_COLI_TIME
    else:
        return f_d - C.NOM_DIST * C.LEN_SCL


def safeAct2(act, eIdC):
    newAct = act
    rstPrfAct = True

    e_ln = eIdC[C.E_LN]
    eg_v = eIdC[C.E_SP]

    # default is lower than the bound, forced accl only if on lane and distance is higher than bound
    cf_d = C.M_ACL_DIST - 1
    cf_v = params.CAR_MIN_SPEED
    if e_ln % 4 == 0:
        if e_ln == params.ROAD_RIGHT_LANE:
            cf_d = eIdC[C.FR_D]
            cf_v = eIdC[C.FR_V]
        elif e_ln == params.ROAD_CENTER_LANE:
            cf_d = eIdC[C.FC_D]
            cf_v = eIdC[C.FC_V]
        elif e_ln == params.ROAD_LEFT_LANE:
            cf_d = eIdC[C.FL_D]
            cf_v = eIdC[C.FL_V]

    # safety check

    # in lane as of now disabled change M_ACL_DIST to get different value
    # just accl as the front car is really far away
    if (e_ln % 4) == 0 and ((act <= C.A_HM and chkSafeDist(eg_v, cf_d, cf_v) <= 0) or (act >= C.A_BL)):
        # in lane and action is maintain, accl or decl check for safety
        # if it leads to collision get a replacement action
        newAct = rplAct(eg_v, cf_d, cf_v)
        return newAct, rstPrfAct
    elif (e_ln % 4) == 0 and cf_d >= C.M_ACL_DIST and\
            (eg_v < params.CAR_MAX_SPEED) and (act != C.A_AM):
        # no front car and not accl hence force accl
        newAct = C.A_AM
        return newAct, rstPrfAct
    elif (e_ln % 4) == 0 and cf_d >= C.M_ACL_DIST and\
            (eg_v == params.CAR_MAX_SPEED) and (act != C.A_MM):
        # no front car and maximum speed
        # maximum speed hence cant accelerate further
        newAct = C.A_MM
        return newAct, rstPrfAct
    elif e_ln <= params.ROAD_RIGHT_LANE:
        # on Right lane
        if act == C.A_BR or act == C.A_MR:
            # not possible to change lane to right
            cf_d = eIdC[C.FR_D]
            cf_v = eIdC[C.FR_V]
            newAct = rplAct(eg_v, cf_d, cf_v)
            return newAct, rstPrfAct
    elif e_ln >= params.ROAD_LEFT_LANE:
        # on left lane
        if act == C.A_BL or act == C.A_ML:
            # not possible to change lane to right
            cf_d = eIdC[C.FL_D]
            cf_v = eIdC[C.FL_V]
            newAct = rplAct(eg_v, cf_d, cf_v)
            return newAct, rstPrfAct

    if act == C.A_ML:
        #  if action is change lane to left check for safe action if it leads to collision
        #  get left car works only for level 0
        if e_ln < C.C_LN:  # on right line
            # left car position and velocity
            fl_d = eIdC[C.FC_D]
            fl_v = eIdC[C.FC_V]
            rl_d = eIdC[C.RC_D]
            rl_v = eIdC[C.RC_V]
            # right car position and velocity + lane number
            fr_d = eIdC[C.FR_D]
            fr_v = eIdC[C.FR_V]
            fr_ln = eIdC[C.FR_P]
        else:
            fl_d = eIdC[C.FL_D]
            fl_v = eIdC[C.FL_V]
            rl_d = eIdC[C.RL_D]
            rl_v = eIdC[C.RL_V]

            fr_d = eIdC[C.FC_D]
            fr_v = eIdC[C.FC_V]
            fr_ln = eIdC[C.FC_P]
        if (chkSafeDist(eg_v, fl_d, fl_v) <= 0) \
                or ((chkSafeDist(eg_v, fr_d, fr_v) <= 0) and abs(e_ln-fr_ln) < C.MIN_LN_SEP)\
                or (chkRearSafeDist(eg_v, rl_d, rl_v) <= 0):
            if (e_ln % 4) != 0 and e_ln >= params.ROAD_RIGHT_LANE:
                # only on lane mark
                newAct = np.random.randint(C.A_BL, params.CAR_NUM_ACTIONS)
                rstPrfAct = True
            else:
                # get alternate action for in lane
                newAct = rplAct(eg_v, cf_d, cf_v)
                rstPrfAct = True
        else:
            newAct = act
            rstPrfAct = False

        return newAct, rstPrfAct

    elif act == C.A_MR:
        # if action is change lane to right check for safe action if it leads to collision
        # get right car works only for level 0
        if e_ln > C.C_LN:
            fr_d = eIdC[C.FC_D]
            fr_v = eIdC[C.FC_V]
            rr_d = eIdC[C.RC_D]
            rr_v = eIdC[C.RC_V]

            fl_d = eIdC[C.FL_D]
            fl_v = eIdC[C.FL_V]
            fl_ln = eIdC[C.FL_P]
        else:
            fr_d = eIdC[C.FR_D]
            fr_v = eIdC[C.FR_V]
            rr_d = eIdC[C.RR_D]
            rr_v = eIdC[C.RR_V]

            fl_d = eIdC[C.FC_D]
            fl_v = eIdC[C.FC_V]
            fl_ln = eIdC[C.FC_P]
        if (chkSafeDist(eg_v, fr_d, fr_v) <= 0) \
                or ((chkSafeDist(eg_v, fl_d, fl_v) <= 0) and (e_ln-fl_ln) < C.MIN_LN_SEP) \
                or (chkRearSafeDist(eg_v, rr_d, rr_v) <= 0):
            if (e_ln % 4) != 0 and e_ln <= params.ROAD_LEFT_LANE:
                newAct = np.random.randint(C.A_BL, params.CAR_NUM_ACTIONS)
                rstPrfAct = True
            else:
                # get alternate action for in lane
                newAct = rplAct(eg_v, cf_d, cf_v)
                rstPrfAct = True
        else:
            newAct = act
            rstPrfAct = False
        return newAct, rstPrfAct

    elif act == C.A_BL:
        newAct = act
        rstPrfAct = False
        return newAct, rstPrfAct

    elif act == C.A_BR:
        newAct = act
        rstPrfAct = False
        return newAct, rstPrfAct

    # figure out best alternate action
    return newAct, rstPrfAct


def safeActEval(act, eIdC, sfLMAct):
    newAct = act
    rstPrfAct = True

    e_ln = eIdC[C.E_LN]
    eg_v = eIdC[C.E_SP]

    # default is lower than the bound, forced accl only if on lane and distance is higher than bound
    cf_d = C.M_ACL_DIST - 1
    cf_v = params.CAR_MIN_SPEED
    if e_ln % 4 == 0:
        if e_ln == params.ROAD_RIGHT_LANE:
            cf_d = eIdC[C.FR_D]
            cf_v = eIdC[C.FR_V]
        elif e_ln == params.ROAD_CENTER_LANE:
            cf_d = eIdC[C.FC_D]
            cf_v = eIdC[C.FC_V]
        elif e_ln == params.ROAD_LEFT_LANE:
            cf_d = eIdC[C.FL_D]
            cf_v = eIdC[C.FL_V]

    # safety check

    # in lane as of now disabled change M_ACL_DIST to get different value
    # just accl as the front car is really far away
    if e_ln % 4 == 0 and ((act <= C.A_HM and chkSafeDist(eg_v, cf_d, cf_v) <= 0) or (act >= C.A_BL)):
        # in lane and action is maintain, accl or decl check for safety
        # if it leads to collision get a replacement action
        newAct = rplAct(eg_v, cf_d, cf_v)
        return newAct, rstPrfAct
    elif (e_ln % 4) == 0 and cf_d >= params.CAR_MAXVIS_DISTANCE and cf_v >= params.CAR_MAX_SPEED and\
            (eg_v < params.CAR_MAX_SPEED) and (act != C.A_AM):
        # no front car and not accl hence force accl
        newAct = C.A_AM
        return newAct, rstPrfAct
    elif (e_ln % 4) == 0 and cf_d >= params.CAR_MAXVIS_DISTANCE and cf_v >= params.CAR_MAX_SPEED and\
            (eg_v == params.CAR_MAX_SPEED) and (act != C.A_MM):
        # no front car and maximum speed
        # maximum speed hence cant accelerate further
        newAct = C.A_MM
        return newAct, rstPrfAct
    elif e_ln == params.ROAD_RIGHT_LANE:
        # on Right lane
        if act == C.A_BR or act == C.A_MR:
            # not possible to change lane to right
            cf_d = eIdC[C.FR_D]
            cf_v = eIdC[C.FR_V]
            newAct = rplAct(eg_v, cf_d, cf_v)
            return newAct, rstPrfAct
    elif e_ln == params.ROAD_LEFT_LANE:
        # on left lane
        if act == C.A_BL or act == C.A_ML:
            # not possible to change lane to right
            cf_d = eIdC[C.FL_D]
            cf_v = eIdC[C.FL_V]
            newAct = rplAct(eg_v, cf_d, cf_v)
            return newAct, rstPrfAct

    if act == C.A_ML:
        # if action is change lane to left check for safe action if it leads to collision
        # get left car works only for level 0
        if e_ln < C.C_LN:
            # left car position and velocity
            fl_d = eIdC[C.FC_D]
            fl_v = eIdC[C.FC_V]

            rl_d = eIdC[C.RC_D]
            rl_v = eIdC[C.RC_V]
            # right car position and velocity + lane number
            fr_d = eIdC[C.FR_D]
            fr_v = eIdC[C.FR_V]
            fr_ln = eIdC[C.FR_P]
        else:
            fl_d = eIdC[C.FL_D]
            fl_v = eIdC[C.FL_V]
            rl_d = eIdC[C.RL_D]
            rl_v = eIdC[C.RL_V]

            fr_d = eIdC[C.FC_D]
            fr_v = eIdC[C.FC_V]
            fr_ln = eIdC[C.FC_P]
        if (chkSafeDist(eg_v, fl_d, fl_v) <= 0) \
                or ((chkSafeDist(eg_v, fr_d, fr_v) <= 0) and abs(e_ln-fr_ln) < C.MIN_LN_SEP)\
                or (chkRearSafeDist(eg_v, rl_d, rl_v) <= 0):
            if isOnLaneMark(e_ln):
                # only on lane mark
                newAct = sfLMAct
                rstPrfAct = True
            else:
                # get alternate action for in lane
                newAct = rplAct(eg_v, cf_d, cf_v)
                rstPrfAct = True
        else:
            newAct = act
            rstPrfAct = False

        return newAct, rstPrfAct

    elif act == C.A_MR:
        # if action is change lane to right check for safe action if it leads to collision
        # get right car works only for level 0
        if e_ln > C.C_LN:
            fr_d = eIdC[C.FC_D]
            fr_v = eIdC[C.FC_V]

            rr_d = eIdC[C.RC_D]
            rr_v = eIdC[C.RC_V]

            fl_d = eIdC[C.FL_D]
            fl_v = eIdC[C.FL_V]
            fl_ln = eIdC[C.FL_P]
        else:
            fr_d = eIdC[C.FR_D]
            fr_v = eIdC[C.FR_V]

            rr_d = eIdC[C.RR_D]
            rr_v = eIdC[C.RR_V]

            fl_d = eIdC[C.FC_D]
            fl_v = eIdC[C.FC_V]
            fl_ln = eIdC[C.FC_P]
        if (chkSafeDist(eg_v, fr_d, fr_v) <= 0) \
                or ((chkSafeDist(eg_v, fl_d, fl_v) <= 0) and (e_ln-fl_ln) < C.MIN_LN_SEP)\
                or (chkRearSafeDist(eg_v, rr_d, rr_v) <= 0):
            if (e_ln % 4) != 0:
                newAct = sfLMAct
                rstPrfAct = False
            else:
                # get alternate action for in lane
                newAct = rplAct(eg_v, cf_d, cf_v)
                rstPrfAct = True
        else:
            newAct = act
            rstPrfAct = False

        return newAct, rstPrfAct

    elif act == C.A_BL:
        newAct = act
        rstPrfAct = False
        return newAct, rstPrfAct
    elif act == C.A_BR:
        newAct = act
        rstPrfAct = False
        return newAct, rstPrfAct

    # figure out best alternate action

    return newAct, rstPrfAct
