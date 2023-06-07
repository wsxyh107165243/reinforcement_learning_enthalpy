'''
    Reinforcemnet learning agent.

    @author:        Xian Yuehui <xianyuehui@stu.xjtu.edu.cn>
    @date:          20220416
    @license:       BSD3
'''
import math
import os
import pickle
import random
from typing import List
import torch
import math
from torch import optim
from DqlModel import DqlModel
from arguments import *
from surrogate import Surrogate
from state import State
from memory import ReplayMemory
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Agent:
    '''
        The reinforcement learning agent should do the following things:
            1. Select an action according to epsilon greedy policy with a dynamically altering parameter.
                - self.__epsilon_start
                - self.__epsilon_decay_coef
                - self.__epsilon_end
                This can be called - take a move.
            2. Standard procedures:
                - fill the buffer with experiences --- Done in ReplayMemory initialization.
                - create a blank start state --- this is simple
                - take an action according to policy_network, epsilon-greedy algorithm
                - shove this transition into replay memory buffer
                - experience replay with a minibatch sampled from buffer
                - memorize (learn) --- update policy network to TD-target network
            3. Internal variables
                - replay memory buffer
                - policy network
                - target network
                - intermediate training results, (turn-on button)
    '''
    def __init__(self, surrogate: Surrogate, replay_memory: ReplayMemory,\
            epsilon_start = EPSILON_START, epsilon_decay_coef = EPSILON_DECAY_COEF, epsilon_end = EPSILON_END,\
            all_possible_actions = ALL_ACTIONS, sample_batch_size = RL_SAMPLE_BATCH_SIZE, \
            gamma = GAMMA, target_update_period = TARGET_UPDATE_PERIOD):
        self.__surrogate = surrogate
        self.__replay_memory = replay_memory
        self.__epsilon_start = epsilon_start
        self.__epsilon_decay_coef = epsilon_decay_coef
        self.__epsilon_end = epsilon_end
        self.__all_possible_actions = all_possible_actions
        self.__sample_batch_size = sample_batch_size
        self.__gamma = gamma
        self.__target_update_period = target_update_period

        self.__training_epoch = 0
        self.__training_step = 0
        
        self.__training_indicators = list()

        self.__need_trainning = None

        # self.__episodes = list()
    
    # Initialize the inteligent's components using default setup.
    def vitalize(self, learning_rate = LEARNING_RATE, dql_agent_path = DQL_AGENT_PATH):
        if not os.path.exists(dql_agent_path):
            # policy network for Q(s,a) is the final target
            self.__policy_network = DqlModel()
            # target network for calculating TD difference
            self.__target_network = DqlModel()
            # unutilized in current version of code
            # self.__loss_function = torch.nn.SmoothL1Loss()
            # if a pre-trainned model does not exist, then a model need to be trainned.
            self.__need_trainning = True
        else:
            '''
                If program enters this branch, it means dql_agent is in evaluation mode,
                and self.__target_network won't be needed.
            '''
            # load previous model
            print('loading previous DQNs agent backup......'.\
                center(OUTPUT_BLOCK_SIZE, '-'))
            self.__policy_network = DqlModel()
            self.__policy_network.load_state_dict(torch.load(dql_agent_path))
            self.__target_network = DqlModel()
            self.__target_network.load_state_dict(torch.load(dql_agent_path))
            # if a pre-trainned model exists, then a model does not need to be trainned.
            self.__need_trainning = False
        # set global optimizer
        self.__optimizer = optim.Adam(self.__policy_network.parameters(), lr = learning_rate)
    
    # reset Agent.__training_epoch as training_epoch else 0
    def reset_training_epoch(self, training_epoch = None):
        self.__training_epoch = 0 if not training_epoch else training_epoch

    # reset Agent.__replay_memory as replay_memory
    def refresh_memory(self, replay_memory: ReplayMemory):
        self.__replay_memory = replay_memory

    # Explicitly set self.__need_training, regardless if DQL_AGENT_PATH exists.
    def need_training(self, need_training = None):
        self.__need_trainning = need_training

    '''
        Return action (Action|float) according to current state,
        based on epsilon-greedy algorithm.
    '''
    def select_action(self, current_state: State, epsilon: float = 0.0) -> ActionType:
        if not epsilon:
            epsilon = self.__epsilon_end + (self.__epsilon_start - self.__epsilon_end) * \
                math.exp(-1. * self.__training_step / self.__epsilon_decay_coef)
        if random.random() < epsilon:
            # select a random action with epsilon_tmp probability
            return current_state.generate_random_action()
        else:
            # greedy selection
            return self.greedy_select(current_state)
    
    # Return maximum possible next_state_action_value according to given state and dqn_network
    def get_max_state_value(self, state: State, network = None) -> float:
        network = network if network else self.__target_network
        state_rl_feature = torch.tensor(state.get_feature()).float()
        comp_min_idx, comp_max_idx = state.get_action_idx_limits()
        index_mask = torch.tensor([[i for i in range(comp_min_idx, comp_max_idx + 1)]])
        max_state_value = network(state_rl_feature).gather(1, index_mask).max(dim = 1).values.item()
        return max_state_value

    # greedily choose from possible composition actions according to previous selected composition design actions
    def greedy_select(self, current_state: State) -> ActionType:
        # .eval() operation is vary IMPORTANT!
        self.__policy_network.eval()
        with torch.no_grad():
            # current_state_rl_feature = torch.tensor(current_state.get_feature()).unsqueeze(dim = 0)
            current_state_rl_feature = torch.tensor(current_state.get_feature()).float()
            comp_min_idx, comp_max_idx = current_state.get_action_idx_limits()
            index_mask = torch.tensor([[i for i in range(comp_min_idx, comp_max_idx + 1)]])
            action_index = self.__policy_network(current_state_rl_feature).gather(1, index_mask)\
                            .max(dim = 1).indices.item() + comp_min_idx
            return self.__all_possible_actions[action_index]

    def train(self, training_epochs = RL_TRAINING_EPOCHS, log_interval = DEFAULT_LOG_INTERVAL):
        # if a pre-trainned model exists
        if not self.__need_trainning:
            return
        # if a pre-trainned model does not exists
        for _ in range(training_epochs):
            self.__training_epoch += 1
            # evaluation mode for target network
            self.__target_network.eval()
            # training mode for policy network
            self.__policy_network.train()
            # prepare a blank start state
            current_state = State(if_init = True)
            # prepare training state indicating parameters
            loss, total_q = None, None
            for _ in range(EPISODE_LEN):
                '''
                    Prepare a new transition to shove into memory buffer.
                '''
                self.__training_step += 1
                action = self.select_action(current_state)
                next_state = State(previous_state = current_state, action = action)
                transition = self.__surrogate.pack_transition(current_state, action, next_state)
                self.__replay_memory.push(transition)
                current_state = next_state
                # experienced replay
                loss, total_q = self.__experience_replay()

            # log training state indicators
            if self.__training_epoch % log_interval == 0:
                print('rl training epoch: {}, loss: {}, total_q: {}'.format(self.__training_epoch, loss, total_q))
            # store training state indicators
            self.__training_indicators.append(TrainingIndicator(self.__training_epoch, loss, total_q))

            # update TD difference network & policy network evaluation
            if self.__training_epoch % self.__target_update_period == 0:
                # memorize learned knowledge
                self.__target_network.load_state_dict(self.__policy_network.state_dict())
            
            # Evaluation code to be appended. Commented at 20220416.
            
    # Experience replay in DQN algorithm.
    def __experience_replay(self):
        # eliminate residual gradient
        self.__optimizer.zero_grad()
        sample_batch = self.__replay_memory.sample(self.__sample_batch_size)
        # batch sub-fields of Transitions
        sample_batch = Transition(*zip(*sample_batch))
        state_batch = sample_batch.current_state
        action_index_batch = [ACTIONS_TO_INDEX_DICT[x] for x in sample_batch.action]
        delayed_reward_batch = sample_batch.delayed_reward
        next_state_batch = sample_batch.next_state
        
        # tensorize sub-item batches
        state_feature_batch = [torch.tensor(state.get_feature()) for state in state_batch]
        action_index_batch = [torch.tensor(action_index).unsqueeze(dim = 0).unsqueeze(dim = 0) for action_index in action_index_batch]
        delayed_reward_batch = [torch.tensor(delayed_reward).unsqueeze(dim = 0).unsqueeze(dim = 0) for delayed_reward in delayed_reward_batch]
        next_state_feature_batch = [torch.tensor(state.get_feature()) for state in next_state_batch]
        # tensor batches -> tensors
        state_feature_batch = torch.cat(state_feature_batch).float()
        action_index_batch = torch.cat(action_index_batch)
        delayed_reward_batch = torch.cat(delayed_reward_batch).float()
        next_state_feature_batch = torch.cat(next_state_feature_batch).float()
        # current result
        state_action_values = self.__policy_network(state_feature_batch).gather(1, action_index_batch)
        '''
            Calculate TD difference

            tensor.max(dim = 1):
                return maximum values and indices
                max operation will squeeze the tensor,
                remember to unsqueeze it for further use

            tensor.detach():
                return a new Variable detached from the current graph
                returned Variable won't need grad forever
        '''
        # next_state_action_values = self.__target_network(next_state_feature_batch).max(dim = 1).values.detach().unsqueeze(dim = 1)
        non_final_mask = torch.tensor([not state.is_end_state() for state in next_state_batch])
        max_next_state_action_values = torch.zeros(len(next_state_batch))
        max_next_state_action_values[non_final_mask] = \
            torch.tensor([self.get_max_state_value(state) for state in next_state_batch if not state.is_end_state()])
        max_next_state_action_values = max_next_state_action_values.unsqueeze(dim = 1)
        '''
            Dimension adjustment maybe needed
        '''
        # compute expected state-action values
        expected_state_action_values = (max_next_state_action_values * self.__gamma) + delayed_reward_batch
        # compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # DQN network optimization
        loss.backward()
        for param in self.__policy_network.parameters():
            param.grad.data.clamp_(min = -1., max = 1.)
        self.__optimizer.step()

        # assemble criterions and return
        total_q = None
        with torch.no_grad():
            total_q = torch.mean(state_action_values).detach()
        
        return loss.item(), total_q.item()

    def save_knowledge(self, knowledge_save_path: str = DQL_AGENT_PATH):
        torch.save(self.__policy_network.state_dict(), knowledge_save_path)

    def save_training_indicators(self, training_indicator_path: str = DQL_TRAINING_INDICATOR_PATH):
        with open(training_indicator_path, 'wb') as f:
            pickle.dump(self.__training_indicators, f)
    
    '''
        Evaluate learned policy by sample one state_action_sequence.

        @return:    state_action_seq
    '''
    def evaluate_knowledge(self, epsilon: float = None):
        # prepare a blank kirigami structure
        current_state = State(if_init = True)
        action = None
        state_action_seq = list()
        # apply action sequence based on greedy policy
        for _ in range(EPISODE_LEN):
            # Change from greedy select -> epsilon select. --20220827
            # action = self.greedy_select(current_state)
            action = self.select_action(current_state = current_state, epsilon = epsilon)
            next_state = State(previous_state = current_state, action = action)
            state_action_seq.append([current_state.get_composition(), action])
            current_state = next_state
        
        state_action_seq.append([current_state.get_composition(), None])
        return state_action_seq

    '''
        Propose next x value to do experiment
    '''
    def propose_next_experiment(self, epsilon: float = None) -> List[float]:
        return self.evaluate_knowledge(epsilon)[-1][0]

def plot_training_indicators(training_indicators: List[TrainingIndicator]):
    epoch, loss, total_q = list(), list(), list()
    for item in training_indicators:
        epoch.append(item.epoch)
        loss.append(item.loss)
        total_q.append(item.total_q)
        # print('rl training epoch: {}, loss: {}, total_q: {}'.format(item.epoch, item.loss, item.total_q))
    plt.figure()
    plt.scatter(epoch, loss, marker = '*', c = 'r', s = 1)
    plt.scatter(epoch, total_q, marker = 'o', c = 'g', s = 1)
    plt.show()

def load_training_indicators(training_indicators_path):
    with open(training_indicators_path, 'rb') as f:
        training_indicators = pickle.load(f)
    return training_indicators