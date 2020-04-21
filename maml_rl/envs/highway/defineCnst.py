# define constants which will called in the learning file
# Related to simulation environment
FC_D = 6
ST_IDX = 0
ACT_IDX = 1
REW_IDX = 2
NXTST_IDX = 3
DN_IDX = 4
M_BUFLEN = 16
LRN_RATE = 1e-4
QN_IDX = 0
TR_IDX = 1
NUM_EP_FIXCAR = 1e4

E_CAR = 0  # ego car id
T_CAR = 1 # target car id

# afind indices
E_LN = 0
E_SP = 1
FL_D = 2
FL_V = 3
FL_P = 4
FC_D = 5
FC_V = 6
FC_P = 7
FR_D = 8
FR_V = 9
FR_P = 10
RL_D = 11
RL_V = 12
RL_P = 13
RC_D = 14
RC_V = 15
RC_P = 16
RR_D = 17
RR_V = 18
RR_P = 19

M_ACL_DIST = 100
C_LN_LT = 8  # less than 8 is center lane
C_LN_RT = 0  # more than 1 is center lane
R_LN_LT = 4  # less than 4 is right lane
L_LN_RT = 4  # more than 4 is left lane

R_LN = 0
C_LN = 4
L_LN = 8

# M--> Maintain; A--> Accelerate; B-->Brake; H--> Hard brake; L-->To left; R-->To right
A_MM = 0
A_AM = 1
A_BM = 2
A_HM = 3
A_ML = 4
A_MR = 5
A_BL = 6
A_BR = 7

#
NUM_CAR_P_LN = 3
NUM_AFID_E = 2
NUM_AFID_R = 3
AFID_XPOS_OFFSET = 9
X_Y_POS_OFFSET = 2
LN_LENGTH = 0.9
Y_COL_LMT = 2.7
X_COL_LMT = 8

MAX_NUM_STUCK = 10
NOM_DIST = 10
SAFE_DIST = 30
SCL_DIST = 20
MIN_LN_SEP = 2

NO_COLI_TIME = 3  # at least for 2 seconds there wont be any collision
LEN_SCL = 1.5  # at least this times length of car is minimum gap between cars
VEL_SCL = 1.5
VEL_OFFSET = 0
