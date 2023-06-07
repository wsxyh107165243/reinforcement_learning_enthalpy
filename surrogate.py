'''
    Gaussian Process Regressor as environment surrogate.

    @author:    Xian Yuehui <xianyeuhui@stu.xjtu.edu.cn>
    @date:      20220415
    @license:   BSD3 clause
'''
# print(__doc__)

import math
from state import State
from typing import List
from arguments import *
from gpr_typedef import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import \
    ConstantKernel, RBF, WhiteKernel
import numpy as np
import pickle


class Surrogate:
    def __init__(self):
        # prepare buffers
        with open(ALL_FEATURE_ENTHALPY_PATH, 'rb') as f:
            self.__feature_enthalpy_buffer = pickle.load(f)
        self.__proposed_exp_points_buffer = []

        # instantiate GPR model
        gpr_kernel = ConstantKernel() * RBF() + WhiteKernel()
        self.__gp_model = GaussianProcessRegressor(kernel = gpr_kernel, n_restarts_optimizer = 10)
        # train a gpr model with initial exp_points
        self.update_gp_model()

    # save Surrogate.__gp_model -> gp_model_path
    def save_gp_model(self, gp_model_path = GP_MODEL_PATH):
        with open(gp_model_path, 'wb') as f:
            pickle.dump(self.__gp_model, f)
    
    '''
        Update gpr model with one input experiment point.
        Push one experiment point into buffers.

        @input:     one ExperimentPoint
        @output:    none
    '''
    def update_gp_model(self, exp_point: ExperimentPoint = None):
        if exp_point:
            # use one experiment point to update gpr model
            # [Ti, Ni, Cu, Hf, Zr]
            feature = self.cal_feature(exp_point[0][0 : 5])
            # enthalpy
            enthalpy = exp_point[1][0]
            # push feature_enthalpy and exp_point -> buffers
            self.__feature_enthalpy_buffer.append([feature, enthalpy])
            self.__proposed_exp_points_buffer.append(exp_point)
        # self.__feature_enthalpy_buffer: [[*atomic_features, enthalpy]...]
        x = [item[0] for item in self.__feature_enthalpy_buffer]
        y = [[item[1]] for item in self.__feature_enthalpy_buffer]
        self.__gp_model.fit(x, y)

    '''
        @input:     mole percentage of [Ti, Ni, Cu, Hf, Zr]
        @output:    expected abs enthalpy of the alloy
    '''
    def predict(self, compositins: List[float]) -> float:
        norm_atomic_features = self.cal_feature(compositins)
        l2_features = np.atleast_2d(norm_atomic_features)
        pred_val, sigma = self.__gp_model.predict(l2_features, return_std = True)       # reture [[predicted_value]]
        return pred_val[0][0], sigma[0]

    '''
        @input:     mole percentage of [Ti, Ni, Cu, Hf, Zr]
        @output:    normalized feature of ['config_entropy', 'diff_atom_radii', 'ea', 'cs']
    '''
    def cal_feature(self, compositions) -> List[float]:
        # calculate abs features
        config_entropy = sum([0. if not c else -c * math.log(c) for c in compositions])
        r_mean = np.array(R_S).dot(np.array(compositions).T)
        diff_r = math.sqrt(sum([compositions[i] * (1 - R_S[i] / r_mean) ** 2 for i in range(len(compositions))]))
        ea = np.array(EA_S).dot(np.array(compositions).T)
        cs = np.array(CS_S).dot(np.array(compositions).T)
        # normalize features
        normalized_features = (np.array([config_entropy, diff_r, ea, cs]) - np.array(EFF_FEATURE_MINS)) \
                                / np.array(EFF_FEATURE_INTERVALS)
        return normalized_features

    '''
        Pack (s, a, r, s') tuple

        @input:     s, a, s'
        @output:    Transition(s, a, r, s')

        @Note:      r: (mean + k * std)
    '''
    def pack_transition(self, current_state: State, action: ActionType, next_state: State) -> Transition:
        # delayed_reward = - (self.predict(next_state.get_composition()) - \
        #     self.predict(current_state.get_composition()))
        # if next_state.is_end_state():
        #     delayed_reward *= 2.
        next_state_pred_mean, next_state_pred_std = self.predict(next_state.get_composition())
        current_state_pred_mead, current_state_pred_std = self.predict(current_state.get_composition())
        k = UCB_K if UCB_ENABLE else 0.
        delayed_reward = - (
            (next_state_pred_mean - k * next_state_pred_std) - \
            (current_state_pred_mead - k * current_state_pred_std)
            )
        return Transition(current_state, action, delayed_reward, next_state)

    '''
        Save experimental and predicted enthalpy into path.
        For data visualization.

        @input:     saving path
    '''
    def save_exp_pred(self, exp_pred_enthalpy_save_path: str = 'exp_pred_enthalpy.txt'):
        with open(exp_pred_enthalpy_save_path, 'wt') as f:
            f.write('# exp_en pred_en std_dev\n')
            for item in self.__feature_enthalpy_buffer:
                feature = item[0]   # []
                exp_enthalpy = item[1]
                # [[...]] -> [[...]], []
                pred_enthalpy, std_dev = self.__gp_model.predict([feature], return_std= True)
                f.write('{}\t{}\t{}\n'.format(exp_enthalpy, pred_enthalpy[0][0], std_dev[0]))