import scipy.io as sio
import os
import time
from copy import deepcopy

from . import utils_sim
from .utils_evalPolicy import evalEpisode
from .params import *
from .utils_attack import *
from .net_DDQN import *
from .attacker import *
import pickle

# for debug
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
This script is for training the attacker
while importing the trained DQN as target car

The target car's policy network need to work with safe action check (match the original evaluation) 
"""

def checkForTrain():
    # check if enough data is available in the memory to train the model
    return len(safeBuf.memory) > C.M_BUFLEN and len(coliBuf.memory) > C.M_BUFLEN


def updateBuffers(collision, coliBuf, safeBuf, s0, act, rew, s1):
    if collision and rew > 0:  # only save responsible cases
        coliBuf.memory.append([s0, act, rew, s1, True])  # check if this works
        with open("collision.json", "wb") as f:
            pickle.dump(coliBuf, f)
    else:
        safeBuf.memory.append([s0, act, rew, s1, False])  # update safe buffer


if __name__ == "__main__":
    '''Initialize policy and model'''
    # tr.manual_seed(999)  # in order to have the same seed
    # random.seed(8765)
    # np.random.seed(9876)  # set random seed

    # Initialize the buffer
    safeBuf = Memory(MemLen=params.SAFEBUF_MAX_LEN)
    # if exist pre-saved collision cases, load it
    try:
        with open("collision.json", "rb") as f:
            coliBuf = pickle.load(f)
    except FileNotFoundError:
        coliBuf = Memory(MemLen=params.COLIBUF_MAX_LEN)

    # set up cuda
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

    # Create two agents
    """ training agent: the attacker """
    attacker_agent = attacker()
    attacker_agent.to(device)
    """ target agent with fixed policy """
    target_agent = DDQN_agent()
    target_agent.to(device)
    # load agent
    pathname_trained = os.path.abspath('out/SavedNW_zsa/QNet.pth.tar')
    saved_net = tr.load(pathname_trained)
    target_agent.Q_eval_net.load_state_dict(saved_net['Q_eval_net'])
    target_agent.Q_eval_net.eval()

    n_C = 0
    EPSILON = params.EPSILON
    trainNW = False

    BM_size = NUM_EPISODES_VAL
    # Benchmarking result histories
    BM_totRewHist = []
    BM_meanRewHist = []
    BM_numColiHist = []
    BM_failureCodeHist = []
    BM_attColiCountHist = []

    # for timing
    sim_time = []
    agent_learn_time = []

    # train history data
    rList = []
    cumRew = []
    numColi = 0

    for i in range(params.NUM_EPISODES):
        '''coloct rollouts on current policy'''
        numCar = np.random.randint(10,20)
        # History for animation in MATLAB
        posHist, visIdHist, actChoice, appliedAction, rewHist = utils_sim.initAniHist(numCar)

        # Initialize all car positions, with target vehilcle near the attacker
        envCars = initCarsAtt(numCar)

        # get initial position store it for animation
        idC, visIdHist[0] = driveFuncs.getAffordInd(envCars)
        attackerState = getAffIndiOfTarget(envCars[0], envCars)  # np.array (24,)

        """ FOR DEBUG """
        debug = False
        if debug:
            debugVis(envCars, visIdHist[0])

        # Store initial position
        driveFuncs.storeCarsPos(envCars, posHist, 0)
        s0Attacker = scaleStateAtt(attackerState)  # get the current scaled state for ego car
        s0Target = driveFuncs.scaleState(idC[C.T_CAR, :])  # get the current scaled state for target car

        # Start of simulation assume no collision #
        collisionTar = False

        # Result variables
        totRew = 0  # total reward per episode
        numRLAct = 0  # total learned actions per episode
        numSafeAct = 0  # total safety controller actions per episode
        numRandAct = 0  # total random actions per episode
        numSwitch = 0  # counter for switches between accelerate/brake

        s0MatPE = []
        qEstMatPE = []
        actAndAplAct = []
        stateidC = []
        allRew = []
        rewParam = []
        velHist = deque(maxlen=3)

        prfActTar = 0

        tmpMinQS = 0
        tmpMaxQS = -1

        for j in range(params.NUM_STEP):
            t0 = time.time()
            if trainNW:
                # only after training is started reduce exploration
                EPSILON = max(np.exp(-n_C / 2e6), 0.2)

            # training agent: which is the attacker
            attacker_agent.Q_eval_net.eval()
            Q, _, _ = attacker_agent.Q_eval_net(Variable(tr.from_numpy(s0Attacker).to(device), requires_grad=False).float()[None, ...])
            actAtt = int(tr.argmax(Q))

            # target agent
            QTar, _, _ = target_agent.Q_eval_net(
                Variable(tr.from_numpy(s0Target).to(device), requires_grad=False).float()[None, ...])
            actTar = int(tr.argmax(QTar))

            """ safe act policy for target car """
            lnTar = idC[C.T_CAR, C.E_LN]
            if driveFuncs.isOnLaneMark(lnTar):
                # On lane mark
                if prfActTar != 0:
                    # Override learned act if we have preferred action for lane changes
                    actTar = prfActTar
            if QTar.cpu().data.numpy()[0, 6] > QTar.cpu().data.numpy()[0, 7]:
                sfLMAct = 6
            else:
                sfLMAct = 7

            newAct, rstPrfAct = driveFuncs.safeActEval(actTar, idC[C.T_CAR, :], sfLMAct)

            if newAct != actTar:
                # saftey controller overrides the first choice, set the original action for collision
                # print('safe action')
                actTar = newAct

            """ Exploration """
            if np.random.random() < EPSILON:
                # chose random action
                # between 0 and 7
                actAtt = np.random.randint(C.A_MM, C.A_BR + 1)

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

            # save data for animation
            s0MatPE.append(s0Attacker)
            qEstMatPE.append(Q.cpu().data.numpy().flatten())

            # store applied action for ego car
            appliedAction[j] = actAtt

            """ Update cars with new actions """
            envCarsOld = envCars
            driveFuncs.setEgoCarAction(envCars[C.E_CAR], actAtt)  # set action for the ego car
            driveFuncs.setEgoCarAction(envCars[C.T_CAR], actTar)  # set action for the target car
            driveFuncs.setAction(envCars[2:], idC[2:, :])  # set actions for env cars
            driveFuncs.appAction(envCars)  # Apply actions

            idC_N, visIdHist[j + 1] = driveFuncs.getAffordInd(envCars)  # get next state for all cars
            attackerState_N = getAffIndiOfTarget(envCars[0], envCars)  # np.array (24,)

            driveFuncs.storeCarsPos(envCars, posHist, j + 1)  # Store position

            """ FOR DEBUG """
            if debug:
                debugVis(envCars, visIdHist[j+1])

            """ 
            Check collision of the target car
            Target car not neccesarily collide into the attacker 
            """
            collisionTar, id4OtherColliCar = checkCollisionID(envCars[C.T_CAR], envCars)
            collisionAtt, id4OtherColliCarAtt = checkCollisionID(envCars[C.E_CAR], envCars)

            # Get new scaled attacker car state, and target car state
            s1Attacker = scaleStateAtt(attackerState_N)
            s1Target = driveFuncs.scaleState(idC_N[C.T_CAR, :])

            if np.any(s1Attacker > 1.2):
                # this code should never happen
                print(s0Attacker, ' ', actAtt)
                print('something wrong here: ', s1Attacker)
                sio.savemat('posHist',
                            {'posHist': posHist, 'numCar': numCar, 'visIdHist': visIdHist,
                             'appliedAction': appliedAction, 'actChoice': actChoice, 'k': j + 1})

                exit()

            """ write reward code here """
            rew, failureCode = rewRespSensAtt(envCars, collisionTar, collisionAtt, id4OtherColliCar)
            """
            failureCode:
            -1: no collision of the target, but the attacker may collide to other env. cars
            0: the other car's fault
            
            1: no car is changing lane; target car is crashed from behind, the other car with evasive action
            
            2: no car is changing lane; target car crash to the front car, with evasive action
            3: no car is changing lane; target car crash to the front car, without evasive action
            
            4: target car is changing lane; target car crash to the tageted lane env. car
            
            5: both car change lane and the target car is the left car who is response to the crash
            6: both car change lane and the target car is the left car who is response to the crash, with evasive action
            
            7: both car change lane and the env car is the left car who is response to the crash, with evasive action
            """

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

            maxQ = tr.max(Q).cpu().data.numpy()
            minQ = tr.min(Q).cpu().data.numpy()  # get minimum Q value
            if tmpMinQS > minQ:
                tmpMinQS = minQ
            if tmpMaxQS < maxQ:
                tmpMaxQS = maxQ

            t1 = time.time()
            sim_time.append(t1-t0)
            """ add rollout (s,a,s') to replay buffer """
            updateBuffers(collisionTar, coliBuf, safeBuf, s0Attacker, actAtt, rew, s1Attacker)

            """ implement deep RL here """
            # get a minibatch size of data from safeBuf and coliBuf
            if not trainNW:
                # check if enough data is available
                trainNW = checkForTrain()
            else:
                # enough data is available so train
                # for every step, it will train once
                stateBuf, actionBuf, rewards, nextStateBuf, doneBuf, \
                stateBuf_array, actionBuf_array, rewards_array, nextStateBuf_array, doneBuf_array\
                    = getSAT(safeBuf, coliBuf, device)
                loss = attacker_agent.learn_step(stateBuf, actionBuf, rewards, nextStateBuf, doneBuf, device)
                t2 = time.time()
                agent_learn_time.append(t2-t1)
                if j == params.NUM_STEP - 1 and i%100==0:
                    print("Simulation time: ", np.mean(sim_time))
                    print("Agent learning step time: ", np.mean(agent_learn_time))
                    print('\t + RL Episode: ', i, ';  RL  step:', j, ';   loss:', loss.cpu().data.numpy())

            # anneal epsilon
            n_C += 1  # steps * episode

            if collisionTar or collisionAtt or (C.T_CAR not in visIdHist[j + 1]):
                break
            else:
                # no collision so continue
                idC, s0Attacker, s0Target = idC_N, s1Attacker, s1Target

        if collisionAtt:
            print('Attacker collision: ', 'the other car id: ', id4OtherColliCarAtt, '; ',
                  'action: ', actAtt, '; rew:', rew)

        if trainNW and i > 0 and i % 20 == 0:
            # every 20 episodes copy the target network
            attacker_agent.update_target_network()

        # CHECK IF CUM REW WORKS
        rList.append(numRLAct)  # append total learned action each episode

        if len(rList) > 100:
            r_mean = np.mean(rList[-100:])  # only calculate recent 100 episode
        else:
            r_mean = np.mean(rList)

        cumRew.append(totRew)  # update the array if numcar was used else use nan
        if len(cumRew) > 100:
            meanRew = np.mean(cumRew[-100:])
        else:
            meanRew = np.mean(cumRew)

        if (i % 100) == 0:
            print('Safe buffer size: ', len(safeBuf.memory))
            print('Collision buffer size: ', len(coliBuf.memory))
            print('Ep:', i, 'numC', numCar,
                  'Episilon: {:.4f}'.format(EPSILON), ' num RL ', numRLAct,
                  ' rd ', numRandAct, ' sf ', numSafeAct, ' tot ', j + 1,
                  'mean num RL {:.2f}'.format(r_mean), ' tot rew {:.2f}'.format(totRew),
                  'mean rew {:.2f}'.format(meanRew),
                  'num coli ', numColi, 'min Q', tmpMinQS, 'max Q', tmpMaxQS)

        if (i % 100) == 0:
            # if i % 1000 == 0:
            #     debugeval = True
            # else:
            #     debugeval = False

            debugeval = False

            BM_envCars = []
            for BM_idx in range(BM_size):
                BM_envCars.append(initCarsAtt(numCar))

            BM_rewHist = np.ones([BM_size, params.NUM_STEP], dtype=np.float) * -10
            BM_numColi = np.ones([BM_size, 1], dtype=np.int) * -1
            BM_failureCode = np.ones([BM_size, params.FAILURE_CODE_NUM], dtype=np.int) * -1
            BM_attColiCount = np.ones([BM_size, params.FAILURE_CODE_NUM], dtype=np.int) * -1

            # Evaluate on benchmark set
            for BM_idx in range(BM_size):
                # use deepcopy so that changes to tmpEnvCars won't change original envCars
                BM_envCars_copy = deepcopy(BM_envCars[BM_idx])

                BM_rewHist[BM_idx, :], BM_numColi[BM_idx], BM_failureCode[BM_idx, :], BM_attColiCount[BM_idx, :] = \
                    evalEpisodeAtt(attacker_agent, target_agent, BM_envCars_copy, i, 0, device, debug=debugeval)

            BM_meanRew = np.mean(BM_rewHist)
            print('\t Episode: ', i, ' mean_rew: ', BM_meanRew)

            # Store benchmark result histories
            BM_totRewHist.append(BM_rewHist)
            BM_meanRewHist.append(BM_meanRew)
            BM_numColiHist.append(BM_numColi)
            BM_failureCodeHist.append(BM_failureCode)
            BM_attColiCountHist.append(BM_attColiCount)

            print('========================================== \n','Episode ', i,  '\n',
                  ' failureCode 0 ', BM_failureCode[-1, 0], '\n',
                  ' failureCode 1 ', BM_failureCode[-1, 1], '\n',
                  ' failureCode 2 ', BM_failureCode[-1, 2], '\n',
                  ' failureCode 3 ', BM_failureCode[-1, 3], '\n',
                  ' failureCode 4 ', BM_failureCode[-1, 4], '\n',
                  ' failureCode 5 ', BM_failureCode[-1, 5], '\n',
                  ' failureCode 6 ', BM_failureCode[-1, 6], '\n',
                  ' failureCode 7 ', BM_failureCode[-1, 7], '\n',
                  ' num coli', BM_numColi[-1], '\n',
                  '========================================== \n')

            BM_filename = os.path.abspath('out/SavedNW_zsa/AttackBenchmarkEvalHist.mat')
            sio.savemat(BM_filename, {'totRewHist': BM_totRewHist,
                                      'meanRewHist': BM_meanRewHist,
                                      'numColiHist': BM_numColiHist,
                                      'failureCodeHist': BM_failureCodeHist,
                                      'attColiCountHist': BM_attColiCountHist})

            train_hist_filename = os.path.abspath('out/SavedNW_zsa/AttackTrainHist.mat')
            sio.savemat(train_hist_filename, {'last Qmax': tmpMaxQS})

        # Save checkpoint
        # if (i > 0) and (i % 10000 == 0):
        if (i > 0) and (i % 1000 == 0):
            fileName = attacker_agent.Q_eval_net.name + '_ckpt_' + str(i) + '.pth.tar'
            qNetdir_ckpt = os.path.abspath('out/SavedNW_zsa/ckptAtt/')
            qNetPathName_ckpt = os.path.join(qNetdir_ckpt, fileName)
            tr.save({
                'Q_eval_net': attacker_agent.Q_eval_net.state_dict(),
                'Q_tar_net': attacker_agent.Q_tar_net.state_dict(),
                'optimizer': attacker_agent.opt.state_dict(),
                'ckpt_episode': i
            }, qNetPathName_ckpt)

    print('Safe buffer size: ', len(safeBuf.memory))
    print('Collision buffer size: ', len(coliBuf.memory))


    # save both Q and target Q network
    qNetPathName = os.path.abspath('out/SavedNW_zsa/AttackQNet.pth.tar')
    tr.save({
        'Q_eval_net': attacker_agent.Q_eval_net.state_dict(),
        'Q_tar_net': attacker_agent.Q_tar_net.state_dict(),
        'optimizer': attacker_agent.opt.state_dict(),
        'ckpt_episode': i
    }, qNetPathName)

    train_hist_filename = os.path.abspath('out/SavedNW_zsa/AttackTrainHist.mat')
    sio.savemat(train_hist_filename, {'last Qmax': tmpMaxQS})

    BM_filename = os.path.abspath('out/SavedNW_zsa/AttackBenchmarkEvalHist.mat')
    sio.savemat(BM_filename, {'totRewHist': BM_totRewHist,
                                      'meanRewHist': BM_meanRewHist,
                                      'numColiHist': BM_numColiHist,
                                      'failureCodeHist': BM_failureCodeHist,
                                      'attColiCountHist': BM_attColiCountHist})

    print('All done and data saved')
