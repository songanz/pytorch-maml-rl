# These are all simulation parameters
NUM_EPISODES = int(1e5)  # a million episodes maybe more
NUM_STEP = 200
NUM_EPISODES_VAL = 100  # use in evaldRLModAct

NUM_STEP_CVAE = 10
CVAE_EPI = 5
ETA = 10  # surprised bonus parameter

# Length of buffer
SAFEBUF_MAX_LEN = int(1e6)
COLIBUF_MAX_LEN = int(1e6)

NUM_SAFE_SAMPLE = int(16)
NUM_COLI_SAMPLE = int(16)
# NUM_DANG_SAMPLE = int(8)
# NUM_LEARN_SAMPLE = NUM_SAFE_SAMPLE+NUM_COLI_SAMPLE+NUM_DANG_SAMPLE
NUM_LEARN_SAMPLE = NUM_SAFE_SAMPLE+NUM_COLI_SAMPLE
BATCH_SIZE = NUM_LEARN_SAMPLE

# For Q-learning update rate and exploration
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1
DDQN = True  # use double dqn or False for DQN

SIM_TIME_STEP = 1
SIM_NUM_STEP = 200
ROAD_NUM_LANES = 2  # total lane 3, starts at 0
SIM_MAX_DISTANCE = 240
ROAD_LANE_WIDTH = 3.6

ROAD_LANE_CENTER = 1.8


ROAD_LEFT_LANE = 8
ROAD_CENTER_LANE = 4
ROAD_RIGHT_LANE = 0

# These are car parameters
CAR_WIDTH = 2
CAR_LENGTH = 6
NUM_VIS_CARS = 6

CAR_MIN_SPEED = 22  # ~50mph
CAR_MAX_SPEED = 32  # ~70mph
CAR_SPD_RANGE = CAR_MAX_SPEED-CAR_MIN_SPEED

# Actions
CAR_ACCEL_RATE = 2
CAR_DECEL_RATE = -2
CAR_HARD_DECEL_RATE = -4
CAR_TO_LEFT = 0.9
CAR_TO_LEFT_LN = 1
CAR_TO_RIGHT = -0.9
CAR_TO_RIGH_LN = -1
CAR_MAINTAIN = 0
ROAD_LN_GAP = 0.9

CAR_REF_DIST = 12
CAR_DISTANCE_0 = 15
CAR_DISTANCE_1 = 30
CAR_DISTANCE_2 = 45
CAR_DISTANCE_3 = 60
CAR_DISTANCE_4 = 120


CAR_VEL_0 = CAR_MIN_SPEED  # 22
CAR_VEL_1 = CAR_VEL_0+CAR_ACCEL_RATE  # 24
CAR_VEL_2 = CAR_VEL_1+CAR_ACCEL_RATE  # 26
CAR_VEL_3 = CAR_VEL_2+CAR_ACCEL_RATE  # 28
CAR_VEL_4 = CAR_VEL_3+CAR_ACCEL_RATE  # 30
CAR_VEL_5 = CAR_VEL_3+CAR_ACCEL_RATE  # 32


CAR_SAFE_DISTANCE = 18  # the minimum seperation between cars for lane change to happen
CAR_MAXVIS_DISTANCE = 120  # is the maximum visible distance between the cars


CAR_NUM_ACTIONS = 8
CAR_NUM_STATES = 20  # 3 per car, 3*3 for car in the front + 3*3 for car in the back + 2 for ego car
CAR_NUM_STATES_ATT = 24  # 3 per car, 6*3 for surrounding vehicle, 3 + 1 (action) for the victim vehicle, 2 for ego car
FAILURE_CODE_NUM = 8  # 0 to 7
CAR_MAX_NUMS = 21
CAR_MIN_NUMS = 10

SURROUNDING_CAR_NUM_ACTIONS = 20  # 2 per car for surrounding vehicle
SURROUNDING_CAR_NUM = 6
