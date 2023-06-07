'''
    Environment state class.

    @author:    Xian Yuehui <xianyuehui@stu.xjtu.edu.cn>
    @date:      20220415
    @license:   BSD3 clause
'''
import random
from copy import deepcopy
from arguments import *

class State:
    def __init__(self, if_init: bool = False, previous_state = None, action: ActionType = None, episode_len = EPISODE_LEN):
        if if_init:
            # atomic fraction (%) of [Ti, Ni, Cu, Hf, Zr]
            self.__composition = [0.5, 0.5, 0, 0, 0]
            self.__episode_len = episode_len
            self.__episode_count = 0
        else:
            self.__composition = deepcopy(previous_state.get_composition())
            previous_episode_no = previous_state.get_episode_count()
            self.__episode_len = episode_len
            self.__episode_count = previous_episode_no + 1
            # composition substitution in the reversed order --pre20220506
            # composition substitution in the forward sequence
            # substitution_index = len(self.__composition) - self.__episode_count
            substitution_index = self.__episode_count
            self.__composition[substitution_index] = action

            '''
                Defining substitution rules:
                    1. Ti's index is 0, Ni's index is 1
                    2. Substitute Ni first, then substitute Ti
                        -- Old version of substitution rule --20220506
            '''
            # if self.__episode_count < episode_len:
            #     if self.__composition[0] + NI_COMP_MIN + sum(self.__composition[2:]) <= 1.0:
            #         self.__composition[1] = 1. - self.__composition[0] - sum(self.__composition[2:])
            #     else:
            #         self.__composition[1] = NI_COMP_MIN
            #         self.__composition[0] = 1. - sum(self.__composition[1:])
            # else:
            #     self.__composition[0] = 1. - sum(self.__composition[1:])
            '''
                Defining substitution rules:
                    1. Ti's index is 0, Ni's index is 1
                    2. Tune Ti:Ni ratio first
                    2. Substitute Ni with Cu first, then substitute Ti with Hf and Zr
                    3. No subsitution needed for Ni
                        -- New version of substitution rule --20220506
            '''
            if substitution_index == 1:
                # tune (Ti + Hf + Zr) : (Ni + Cu) ratio
                self.__composition[0] = 1.0 - self.__composition[1]
            elif substitution_index == 2:
                # substitute Ni with Cu
                self.__composition[1] -= self.__composition[2]
            elif substitution_index > 2:
                # Substitute Ti with Hf and Zr
                self.__composition[0] = 1. - sum(self.__composition[1:])

            # round up compositions
            for idx in range(len(self.__composition)):
                self.__composition[idx] = round(self.__composition[idx], \
                    COMPOSITION_ROUNDUP_DIGITS)

    def get_episode_len(self) -> int:
        return self.__episode_len

    def get_episode_count(self) -> int:
        return self.__episode_count
    
    def get_composition(self):
        return self.__composition

    # len(feature) corresponds to flattened dimensions in DqlModel.
    def get_feature(self):
        feature = deepcopy(self.__composition)
        feature.append(self.__episode_count)
        return feature

    def is_end_state(self):
        return self.__episode_count == self.__episode_len

    '''
        Get action limits according to current state.

        @output:    (composition_min_idx, composition_max_idx)
                        composition_min_idx * COMPOSITION_INTERVAL == composition_lower_limit_in_float
                        composition_max_idx * COMPOSITION_INTERVAL == composition_upper_limit_in_float
    '''
    def get_action_idx_limits(self):
        # old version of comp_max and comp_min --pre20220506
        # elem_index = self.__episode_len - self.__episode_count
        # comp_max = min(COMP_MAX_LIMITS[elem_index], \
        #     1.0 - sum(self.__composition[elem_index + 1:]) - sum(COMP_MIN_LIMITS[:elem_index]))
        # comp_min = max(COMP_MIN_LIMITS[elem_index], \
        #     1.0 - sum(self.__composition[elem_index + 1:]) - sum(COMP_MAX_LIMITS[:elem_index]))
        # new version of comp_max and comp_min --20220506
        elem_index = self.__episode_count + 1
        if elem_index == 1:
            comp_min, comp_max = COMP_MIN_LIMITS[elem_index], COMP_MAX_LIMITS[elem_index]
        elif elem_index == 2:
            comp_min = COMP_MIN_LIMITS[elem_index]
            comp_max = min(COMP_MAX_LIMITS[elem_index], self.__composition[1])
        else:
            comp_min = max(COMP_MIN_LIMITS[elem_index], 
                1.0 - sum(self.__composition[1 : elem_index]) \
                    - sum(COMP_MAX_LIMITS[elem_index + 1:]) \
                    - COMP_MAX_LIMITS[0])
            comp_max = min(COMP_MAX_LIMITS[elem_index],
                1.0 - sum(self.__composition[1 : elem_index]) \
                    - sum(COMP_MIN_LIMITS[elem_index + 1:]) \
                    - COMP_MIN_LIMITS[0])
        comp_min_idx = ACTIONS_TO_INDEX_DICT[round(comp_min, COMPOSITION_ROUNDUP_DIGITS)]
        comp_max_idx = ACTIONS_TO_INDEX_DICT[round(comp_max, COMPOSITION_ROUNDUP_DIGITS)]
        return comp_min_idx, comp_max_idx

    '''
        Generate one random action that can be applied to this state

        @output:    a random action in float
    '''
    def generate_random_action(self) -> ActionType:
        comp_min_idx, comp_max_idx = self.get_action_idx_limits()
        rand_comp_idx = random.randint(comp_min_idx, comp_max_idx)
        return ALL_ACTIONS[rand_comp_idx]