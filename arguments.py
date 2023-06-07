'''
    General constant definition and data structure definition for 
    DQN algorithm of alloy composition design.

    @author:    Xian Yuehui <xianyuehui@stu.xjtu.edu.cn>
    @date:      20220415
    @licence:   BSD3 clause
'''
from collections import namedtuple

# Genaral data structure definition
Transition = namedtuple('Transition', ('current_state', 'action', 'delayed_reward', 'next_state'))
TrainingIndicator = namedtuple('TrainingIndicator', ('epoch', 'loss', 'total_q'))
CompositionLimit = namedtuple('CompositionLimit', ('min_bound', 'max_bound'))

OUTPUT_BLOCK_SIZE = 100

# surrogate.py
GP_MODEL_PATH = 'gp_model.pk'
EXP_BUFFER_PK_PATH = 'exp_buffer.pk'
DATA_PK_PATH = 'selected_data.pk'
ALL_FEATURE_ENTHALPY_PATH = 'data_feature_enthalpy.pk'
# atomic properties of different elments, [Ti, Ni, Cu, Hf, Zr, Nb, Co, Cr, Fe, Mn, Pd]
# ea (valence electron numbers of average atomic number)
EA_S_ALL = (0.181818182, 0.357142857, 0.379310345, 0.055555556, 0.1, 0.12195122, 0.333333333, 0.25, 0.307692308, 0.28, 0.217391304)
# cs (Pettifor chemical scale)
CS_S_ALL = (0.79, 1.09, 1.2, 0.775, 0.76, 0.82, 1.04, 0.89, 0.99, 0.94, 1.12)
# mr (metallic radius)
# MR_S = (176, 149, 145, 208, 206, 145, 125, 129, 126, 137, 137) # Old MR_S, which is wrong
MR_S_ALL = (147, 125, 128, 159, 160, 145, 125, 129, 126, 137, 137)  # New MR_S, Bug_fix-20220509
# arc (Clementi’s atomic radii)
ARC_S_ALL = (176, 149, 145, 208, 206, 198, 152, 166, 156, 161, 169)
# setting needed atomic features
ELEMENT_COUNT = 5
EA_S = EA_S_ALL[0 : ELEMENT_COUNT]
CS_S = CS_S_ALL[0 : ELEMENT_COUNT]
MR_S = MR_S_ALL[0 : ELEMENT_COUNT]
ARC_S = ARC_S_ALL[0 : ELEMENT_COUNT]
# setting R in diff_r (delta.R, difference of atomic radii) calculation
R_S = MR_S
# parameters for atomic feature normalization
EFF_CONF_S_MIN, EFF_CONF_S_MAX = (0.6931066800131775, 1.499120250668105)
EFF_DIFF_R_MIN, EFF_DIFF_R_MAX = (0.0569916810266724, 0.10777067951565188)
EFF_EA_MIX, EFF_EA_MAX = (0.21990096858000002, 0.27391401709999996)
EFF_CS_MIN, EFF_CS_MAX = (0.9092500000000001, 0.962)
EFF_FEATURE_MINS = (EFF_CONF_S_MIN, EFF_DIFF_R_MIN, EFF_EA_MIX, EFF_CS_MIN)
EFF_FEATURE_INTERVALS = (EFF_CONF_S_MAX - EFF_CONF_S_MIN,
                        EFF_DIFF_R_MAX - EFF_DIFF_R_MIN,
                        EFF_EA_MAX - EFF_EA_MIX,
                        EFF_CS_MAX - EFF_CS_MIN)

# UCB (upper confidence bound)
UCB_ENABLE = False
UCB_K = 0.33

# state.py
ActionType = float
EPISODE_LEN = 4
REPLAY_MEMORY_PATH = 'replay_memory_buffer.pk'
MEMORY_CAPACITY = 3000
RESUME_MEMORY_BUFFER = False
COMPOSITION_INTERVAL = 0.001
COMPOSITION_ROUNDUP_DIGITS = 4
# (Ni + Cu) : [0.47, 0.53], (Ti + Hf + Zr) : [0.47, 0.53], Cu: [0, 0.15], Hf: [0, 0.15], Zr: [0, 0.15]
# [NI_COMP_MIN, NI_COMP_MAX] to define (Ni + Cu) : (Ti + Hf + Zr) ratio, NOT REAL Ni comp constraints
# TI_COMP_MIN needs modification! --20220511
TI_COMP_MIN, TI_COMP_MAX = 0., 0.53
NI_COMP_MIN, NI_COMP_MAX = 0.47, 0.53
CU_COMP_MIN, CU_COMP_MAX = 0., 0.15
HF_COMP_MIN, HF_COMP_MAX = 0., 0.15
ZR_COMP_MIN, ZR_COMP_MAX = 0., 0.15

COMP_LIMITS = (CompositionLimit(TI_COMP_MIN, TI_COMP_MAX),
                    CompositionLimit(NI_COMP_MIN, NI_COMP_MAX),
                    CompositionLimit(CU_COMP_MIN, CU_COMP_MAX),
                    CompositionLimit(HF_COMP_MIN, HF_COMP_MAX),
                    CompositionLimit(ZR_COMP_MIN, ZR_COMP_MAX))

COMP_MIN_LIMITS = CompositionLimit(*zip(*COMP_LIMITS)).min_bound
COMP_MAX_LIMITS = CompositionLimit(*zip(*COMP_LIMITS)).max_bound

# agent.py
EPSILON_START = 0.8
EPSILON_DECAY_COEF = 10000
EPSILON_END = 0.1
LEARNING_RATE = 5e-4            # Modification needed!
RL_TRAINING_EPOCHS = 1000       # Modification needed!
DEFAULT_LOG_INTERVAL = 1000     # terminal log every this epochs
RL_SAMPLE_BATCH_SIZE = 64
GAMMA = 0.80
TARGET_UPDATE_PERIOD = 10
DQL_AGENT_PATH = 'dql_agent.pt'
DQL_TRAINING_INDICATOR_PATH = 'rl_agent_training_indicators.pk'
# composition tuning limits
# LOW_BOUND = 0
COMP_LOW_BOUND_INT = round(min(COMP_MIN_LIMITS) / COMPOSITION_INTERVAL)
COMP_HIGH_BOUND_INT = round(max(COMP_MAX_LIMITS) / COMPOSITION_INTERVAL)
COMP_LOW_BOUND = COMP_LOW_BOUND_INT * COMPOSITION_INTERVAL
COMP_HIGH_BOUND = COMP_HIGH_BOUND_INT * COMPOSITION_INTERVAL
# action definition
ALL_ACTIONS = [round(x * COMPOSITION_INTERVAL, COMPOSITION_ROUNDUP_DIGITS) \
                for x in range(COMP_LOW_BOUND_INT, COMP_HIGH_BOUND_INT + 1)]
ALL_ACTIONS_COUNT = len(ALL_ACTIONS)
ACTIONS_TO_INDEX_DICT = dict(zip(ALL_ACTIONS, range(ALL_ACTIONS_COUNT)))

# DqlModel.py
POSSIBLE_ACTION_COUNT = COMP_HIGH_BOUND_INT - COMP_LOW_BOUND_INT + 1
DROPOUT_PROBABILITY = 0.5
FEATURE_FLATTENED_DIMENSIONS = 6
FIRST_LAYER_IN_FEATURES = FEATURE_FLATTENED_DIMENSIONS
FIRST_LAYER_OUT_FEATURES = 512
SECOND_LAYER_IN_FEATURES = FIRST_LAYER_OUT_FEATURES
SECOND_LAYER_OUT_FEATURES = 256
THIRD_LAYER_IN_FEATURES = SECOND_LAYER_OUT_FEATURES
THIRD_LAYER_OUT_FEATURES = POSSIBLE_ACTION_COUNT

# executer.py
COMP_PROP_PER_ITER = 8

# collision_detector.py
EXISTING_COMP_DATA_PATH = 'existing_comp_data.pk'