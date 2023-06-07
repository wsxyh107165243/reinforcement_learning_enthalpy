'''
    DQN algorithm executer for alloy composition design with high enthalpy change.

    @author:    Xian Yuehui <xianyuehui@stu.xjtu.edu.cn>
    @date:      20220417
    @licence:   BSD3 clause
'''
print(__doc__)

from datetime import datetime
import pickle
import random
from arguments import COMP_PROP_PER_ITER, DQL_AGENT_PATH, EXP_BUFFER_PK_PATH, GP_MODEL_PATH
from surrogate import Surrogate
from memory import ReplayMemory
from agent import Agent
from collision_detector import CollisionDetector
# from gpr_typedef import *

'''
    Main program.
'''
def execute():
    # -------------------------------------------------------------------------
    # parameters setup
    replay_memory_capacity = 5000
    training_epochs = 30000
    print('Repaly memory capacity: {}'.format(replay_memory_capacity))
    print('training epochs: {}'.format(training_epochs))
    
    # -------------------------------------------------------------------------
    # prepare Gaussian Process Regression model for environment surrogate
    surrogate = Surrogate()

    # -------------------------------------------------------------------------
    # instantiate DQN agent
    replay_memory = ReplayMemory(surrogate, capacity = replay_memory_capacity)
    agnt = Agent(surrogate, replay_memory)
    agnt.vitalize()

    # every (trainging_epochs // proposition_logs) per composition proposition.
    proposition_logs = 100
    proposition_log_res = list()

    # -------------------------------------------------------------------------
    # train DQN agent
    for _ in range(proposition_logs):
        # set need_training explicitly
        agnt.need_training(True)

        # train DQN with desired epochs
        agnt.train(training_epochs = training_epochs // proposition_logs)

        # save training details
        # agnt.save_training_indicators()

        # Knowledge evaluation
        print('Knowledge evaluation:')
        proposed_composition = agnt.propose_next_experiment()
        pred_enthalpy = surrogate.predict(proposed_composition)
        print('Proposed composition [Ti, Ni, Cu, Hf, Zr]: {} with predicted enthalpy of {}'.\
                format(proposed_composition, pred_enthalpy))

        proposition_log_res.append((proposed_composition, pred_enthalpy))

    # propose N compositions
    now = datetime.now()
    cd = CollisionDetector()
    with open('proposed compositions-{}-{}-{}.txt'.format(now.year, now.month, now.day), 'wt') as f:
        prop_comp_count = 0
        prop_comp_bucket = []
        while prop_comp_count < COMP_PROP_PER_ITER:
            proposed_composition = agnt.propose_next_experiment(epsilon = 0.1)
            if not cd.collided(proposed_composition):
                prop_comp_count += 1
                cd.update(proposed_composition)
                pred_enthalpy = surrogate.predict(proposed_composition)
                prop_comp_bucket.append((proposed_composition, pred_enthalpy))
        prop_comp_bucket.sort(key = lambda x: x[-1], reverse = True)
        for proposed_composition, pred_enthalpy in prop_comp_bucket:
            f.write('Proposed composition [Ti, Ni, Cu, Hf, Zr]: {} with predicted enthalpy of {}\n'.\
                format(proposed_composition, pred_enthalpy))

    # save trained DQN model
    agnt.save_knowledge(DQL_AGENT_PATH)
    surrogate.save_gp_model(GP_MODEL_PATH)
    
    # -------------------------------------------------------------------------
    # save DQN agent proposition details
    with open('propositon_log_result.pk', 'wb') as f:
        pickle.dump(proposition_log_res, f)

if __name__ == '__main__':
    execute()