'''
    DQN network.

    @architecture:  multi-perceptron
                        1st and 2nd layer:  nonlinear
                        3rd layer:          linear
                    dimensions:    6 -> 512 -> 256 -> action

    @author:        Xian Yeuhui<xianyueui@stu.xjtu.edu.cn>
    @date:          20220416
    @license:       BSD3 clause
'''
import torch
from arguments import *

class DqlModel(torch.nn.Module):
    def __init__(self):
        super(DqlModel, self).__init__()

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features = FIRST_LAYER_IN_FEATURES,
                out_features = FIRST_LAYER_OUT_FEATURES
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = DROPOUT_PROBABILITY)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features = SECOND_LAYER_IN_FEATURES,
                out_features = SECOND_LAYER_OUT_FEATURES
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = DROPOUT_PROBABILITY)
        )
        self.fc3 = torch.nn.Linear(
            in_features = THIRD_LAYER_IN_FEATURES,
            out_features = THIRD_LAYER_OUT_FEATURES
        )
    
    def forward(self, x):
        # linearize
        x = x.view(-1, FIRST_LAYER_IN_FEATURES)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x