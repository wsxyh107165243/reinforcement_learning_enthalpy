'''
    Monte Carlo Sampling (MCS) for the performance
    of Gaussian Process Regressor (GPR) model.

    @author:    Xian Yuehui <xianyuehui@stu.xjtu.edu.cn>
    @date:      20220512
    @licence:   BSD3 clause
'''
# print(__doc__)

import math
import pickle
import os
import random
from typing import List
from arguments import *
from surrogate import Surrogate
from state import State

MSC_COUNT = int(1e7)
BLOCK_SIZE = int(1e4)

def get_all_possible_actions(state: State) -> List[ActionType]:
    comp_min_idx, comp_max_idx = state.get_action_idx_limits()
    return ALL_ACTIONS[comp_min_idx: comp_max_idx + 1]

def mcs(composition_buffer: list, state: State) -> None:
    if state.is_end_state():
        composition_buffer.append(state.get_composition())
    else:
        all_possible_actions = get_all_possible_actions(state)
        random_action = random.choice(all_possible_actions)
        state = State(previous_state = state, action = random_action)
        mcs(composition_buffer, state)

if __name__ == '__main__':
    # ---------------------------------------------------------------
    # mcs sampling
    mcs_compositions = []
    for _ in range(MSC_COUNT):
        blank_state = State(if_init = True)
        mcs(mcs_compositions, blank_state)
        
        if len(mcs_compositions) % BLOCK_SIZE == 0:
            print(len(mcs_compositions) // BLOCK_SIZE)
    
    # ---------------------------------------------------------------
    # calculate features
    surrogate = Surrogate()
    mcs_features = [surrogate.cal_feature(comp) for comp in mcs_compositions]

    # ---------------------------------------------------------------
    # calculate predicted enthalpy
    new_gp_model_path = './models/20220828/gp_model.pk'
    with open(new_gp_model_path, 'rb') as f:
        gp_model = pickle.load(f)

    pred_H_s = []
    for i in range(math.ceil(len(mcs_features) / BLOCK_SIZE)):
        block_features = mcs_features[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
        pred_H_s.extend(gp_model.predict(block_features))

    # ---------------------------------------------------------------
    # pack and sort composition and predicted enthalpy
    mcs_exp_points = [[comp, en[-1]] for comp, en in zip(mcs_compositions, pred_H_s)]
    mcs_exp_points.sort(key = lambda x: x[-1], reverse = False)

    # -------------------------------------------------------------------
    # print some selected result
    selected_idxs = [10 ** i if i else 0 for i in range(8)]
    for idx in selected_idxs:
        if idx < len(mcs_exp_points):
            for comp in mcs_exp_points[idx][0]:
                print('{}\t'.format(comp), end = '')
            print('{}'.format(mcs_exp_points[idx][-1]))

    # -------------------------------------------------------------------
    # get enthalpy probability distribution of mcs compositions
    prob_dict = {}
    for exp_point in mcs_exp_points:
        if exp_point[-1] < 0.:
            key = round(exp_point[-1])
            if key not in prob_dict.keys():
                prob_dict[key] = 1
            else:
                prob_dict[key] += 1

    # -------------------------------------------------------------------
    # total mcs composition count
    total_count = 0
    for v in prob_dict.values():
        total_count += v
    print('total: {}'.format(total_count))

    # -------------------------------------------------------------------
    # print enthalpy probability distribution
    for k, v in prob_dict.items():
        print(k, ' ', v / total_count)    
    





