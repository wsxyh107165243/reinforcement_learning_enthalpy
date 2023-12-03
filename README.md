Reinforcement learning is applied to the compositional design of TiNiCuHfZr alloy to achieve high transformational enthalpy.
    -- version 1.0, 20220416

Files:
    data.txt:           Data containing all alloy compositions and enthalpy values.
    gpr_ver1.py:        Module for obtaining regression models using Gaussian Process Regression (GPR).
    gpr_demo.py：       A demo for using GPR.
    gpr_typedef.py:     Module for defining input-output data structure in GPR.
    gp_model.pk:        A buffered GPR model.
    arguments.py:       The file for defining global variables.
    state.py:           State class for alloy compositional design process.
    surrogate.py:       Environmental surrogate class.
    memory.py:          Memory buffer class.
    DqlModel.py:        Neural network structure for the reinforcement learning agent.
    agent.py:           Implementation for the Deep Q-Network (DQN) algorithm.
    executer.py:        Main program.
    selected_data.pk    Serialized data in the initial dataset.
    gen_all_feature_enthalpy.py     Logics for generating the serialized file of atomic feature data.
    data_feature_enthalpy.pk        Serialized file of atomic features and enthalpy values.
    
Dependencies:
    numpy, matplotlib, torch, sklearn

    To install a python package, suppose <numpy>, run in terminal:
        pip install numpy
    for more info, contact <xianyeuhui@stu.xjtu.edu.cn>

To run the scripts：
    Execute the following command in the folder:
        python executer.py
