import numpy as np

from . import params
from . import defineCnst as C

# See Car.java code by Nan

class Car:
    def __init__(self, id, xvel=0, xPos=0.0, yPos=0.0, laneNum=0,MaxVel = params.CAR_MAX_SPEED,MinVel=params.CAR_MIN_SPEED):
        """
        Details of this function goes here:
        Create a car
        """
        self.id = id
        self.xPos = xPos
        self.xVel = xvel
        self.yPos = yPos
        self.laneNum = laneNum
        self.xVelAct = 0.0  # currently don't set any action while instantiation
        self.yPosAct = 0.0
        self.laneNumAct = 0
        self.ActNum = int(0)  # currently maintain
        self.MaxVel = MaxVel  # by default maximum speed
        self.MinVel = MinVel  # by default minimum speed

    # def setAction(self, xVelAct = 0.0,yPosAct = 0.0,laneNumAct = 0,ActNum=0):
    #     """
    #     Sets action for the given car
    #     """
    #     self.xVelAct =xVelAct
    #     self.yPosAct =yPosAct
    #     self.laneNumAct=laneNumAct
    #     self.ActNum = ActNum

    def stateUpdate(self, egoCarVel):
        """
        Update system states using linear model

        """
        # Set velocity first
        self.xVel += self.xVelAct
        # check and set velocity within threshold
        if self.xVel < self.MinVel:
            self.xVel = self.MinVel   # lower than minimum set to minimum
        if self.xVel > self.MaxVel:
            self.xVel = self.MaxVel   # greater than maximum speed hence limit to maximum
        if self.id == 0:
            self.xPos = 0  # for car 0 i.e., ego car there is no change in its poistion
        else:
            self.xPos += (self.xVel-egoCarVel)*params.SIM_TIME_STEP  # everything moves with respect to ego car
        # check if car is within the bound else warp around
        if abs(self.xPos) > params.SIM_MAX_DISTANCE:
            # update based on sign only if the value is more than bound
            self.xPos -= np.sign(self.xPos)*2*params.SIM_MAX_DISTANCE

        # update y position and lane numbers
        self.yPos += self.yPosAct
        self.laneNum += self.laneNumAct

    def getAffIndiOfaCar(self, envCars):
        """This gives affordance indicator for the current car"""
        # assume everything is far and going away
        frontOffSet = 6
        rearOffSet = -6
        offsetV = 0
        fl_x, fc_x, fr_x = params.CAR_DISTANCE_4, params.CAR_DISTANCE_4, params.CAR_DISTANCE_4
        rl_x, rc_x, rr_x = -1 * params.CAR_DISTANCE_4, -1 * params.CAR_DISTANCE_4, -1 * params.CAR_DISTANCE_4
        # travelling at maximum velocity
        fl_v, fc_v, fr_v = params.CAR_MAX_SPEED+offsetV, params.CAR_MAX_SPEED+offsetV, params.CAR_MAX_SPEED+offsetV
        # travelling at maximum velocity
        rl_v, rc_v, rr_v = params.CAR_MIN_SPEED-offsetV, params.CAR_MIN_SPEED-offsetV, params.CAR_MIN_SPEED-offsetV
        fl_y, fc_y, fr_y = C.L_LN, C.C_LN, C.R_LN
        rl_y, rc_y, rr_y = C.L_LN, C.C_LN, C.R_LN
        fr_id, fc_id, fl_id, rl_id, rc_id, rr_id = -1, -1, -1, -1, -1, -1

        for i in range(len(envCars)):
            if self.id != envCars[i].id:
                # not same car loop around
                rel_x = envCars[i].xPos - self.xPos  # this is distance between the centers of the car
                rel_vel = envCars[i].xVel    # actual velocity
                rel_y = envCars[i].laneNum   # actual y position
                if C.C_LN_RT < rel_y < C.C_LN_LT:
                    # center lane
                    if 0 <= rel_x < fc_x:
                        # front car
                        fc_x = rel_x
                        fc_v = rel_vel
                        fc_y = rel_y
                        fc_id = i
                    elif rc_x < rel_x < 0:
                        # rear car
                        rc_x = rel_x
                        rc_v = rel_vel
                        rc_y = rel_y
                        rc_id = i
                if C.L_LN_RT < rel_y:
                    # left lane
                    if 0 <= rel_x < fl_x:
                        # front car
                        fl_x = rel_x
                        fl_v = rel_vel
                        fl_y = rel_y
                        fl_id = i
                    elif rl_x < rel_x < 0:
                        # rear car
                        rl_x = rel_x
                        rl_v = rel_vel
                        rl_y = rel_y
                        rl_id = i
                if rel_y < C.R_LN_LT:
                    # right lane
                    if 0 <= rel_x < fr_x:
                        # front car
                        fr_x = rel_x
                        fr_v = rel_vel
                        fr_y = rel_y
                        fr_id = i
                    elif rr_x < rel_x < 0:
                        # rear car
                        rr_x = rel_x
                        rr_v = rel_vel
                        rr_y = rel_y
                        rr_id = i
        # car lane and its velocity then in a cyclic manner rest of 5 cars distance and velocities
        afforIndOfCar = np.array([self.laneNum, self.xVel,
                                  fl_x, fl_v, fl_y,
                                  fc_x, fc_v, fc_y,
                                  fr_x, fr_v, fr_y,
                                  rl_x, rl_v, rl_y,
                                  rc_x, rc_v, rc_y,
                                  rr_x, rr_v, rr_y])
        idCars = np.array([fl_id, fc_id, fr_id, rl_id, rc_id, rr_id])
        return afforIndOfCar, idCars
