'''
    Existing data collision detetor.

    @author:    Xian Yuehui <xianyuehui@stu.xjtu.edu.cn>
    @date:      20220828
    @licence:   BSD3 clause
'''
import pickle
from typing import List
from arguments import *

class CollisionDetector:
    def __init__(self, existing_data_path: str = EXISTING_COMP_DATA_PATH):
        with open(existing_data_path, 'rb') as f:
            existing_comp_data = pickle.load(f)

        self.__collision_buffer = set()
        for comp in existing_comp_data:
            self.__collision_buffer.add('-'.join([str(c) for c in comp]))

    # collision detector
    def collided(self, comp: List[float]) -> bool:
        return '-'.join([str(c) for c in comp]) in self.__collision_buffer

    # update existing composition data
    def update(self, comp: List[float]) -> None:
        self.__collision_buffer.add('-'.join([str(c) for c in comp]))