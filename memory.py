'''
    Ring memory buffer for DQN

    @author:    Xian Yuehui <xianyuehui@stu.xjtu.edu.cn
    @date:      20220416
    @licence:   BSD3 clause
'''
from typing import List
from arguments import *
from surrogate import Surrogate
from state import State
import pickle
import random
import os

class ReplayMemory:
    def __init__(self, surrogate: Surrogate, capacity: int = MEMORY_CAPACITY,\
            replay_memory_path: str = REPLAY_MEMORY_PATH, \
            resume_memory_buffer: bool = RESUME_MEMORY_BUFFER):
        self.__surrogate = surrogate
        self.__capacity = int(capacity)
        self.__replay_memory_path = replay_memory_path
        self.__memory_buffer = list()
        # self.__init_trans_count = INITIAL_EXPERIMENTED_COUNT * TRANSITION_PER_EXPERIMNET
        self.__index_adder = 0
        self.__init_memory_buffer(resume_memory_buffer = resume_memory_buffer)

    '''
        Push a transition into the buffer.

        @input:     transition
        @output:    None
    '''
    def push(self, transition: Transition):
        if len(self.__memory_buffer) < self.__capacity:
            self.__memory_buffer.append(transition)
        else:
            # ring buffer (self.__memory_buffer[self.__init_trans_count, ~]) for low fidelity samples
            self.__memory_buffer[self.__index_adder] = transition
            self.__index_adder = (self.__index_adder + 1) % self.__capacity

    '''
        Randomly sample a list of Transition from buffer.

        @input:     sample size
        @output:    a list of Transitions
    '''
    def sample(self, sample_size) -> List[Transition]:
        return random.sample(self.__memory_buffer, sample_size)

    # save current memory into memory_save_path
    def save_current_memory(self, memory_save_path: str = None) -> None:
        if not memory_save_path:
            memory_save_path = self.__replay_memory_path
        with open(memory_save_path, 'wb') as f:
            pickle.dump(memory_save_path, f)

    '''
        Init replay_memory_buffer with few experiences.
    '''
    def __init_memory_buffer(self, resume_memory_buffer):
        if os.path.exists(self.__replay_memory_path) and resume_memory_buffer:
            # print('loading previous memory buffer backup'.center(OUTPUT_BLOCK_SIZE, '-'))
            with open(self.__replay_memory_path, 'rb') as f:
                self.__memory_buffer = pickle.load(f)
        else:
            # print('Generating initial memory buffer'.center(OUTPUT_BLOCK_SIZE, '-'))
            tmp_count = 0
            # generate random transitions
            while tmp_count < self.__capacity:
                # init a blank state
                current_state = State(if_init = True)
                '''
                    For one episode, repeat:
                        1. select a random action
                        2. get next state by applying action to current state
                        3. shove the packed transition into memory_buffer
                        4. current_state <- next_state to prepare for next transition
                        5. tmp_count += 1 to jump out memory initialization
                '''
                for _ in range(EPISODE_LEN):
                    random_action = current_state.generate_random_action()
                    next_state = State(previous_state = current_state, action = random_action)
                    self.push(self.__surrogate.pack_transition(current_state, random_action, next_state))
                    current_state = next_state
                    tmp_count += 1
            
            # save memory buffer, for initialization may be time consuming
            self.save_current_memory()