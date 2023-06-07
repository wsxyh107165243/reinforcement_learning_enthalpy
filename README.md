强化学习应用于TiNiCuHfZr合金成分设计，以实现高焓变。
    -- version 1.0, 20220416

文件介绍:
    data.txt:           包含所有合金成分及焓变的数据
    gpr_ver1.py:        使用<高斯过程回归>(GPR)得到回归模型的模块
    gpr_demo.py：       使用GPR的例程
    gpr_typedef.py:     GPR中用于输入输出数据结构定义的模块
    gp_model.pk:        缓存的GPR模型
    arguments.py:       全局常量统一定义文件
    state.py:           合金成分设计过程中状态描述类
    surrogate.py:       环境替代，即为GPR模型的包装类
    memory.py:          DQN算法的往期记忆缓冲模块
    DqlModel.py:        状态->动作预期回报的神经网络网络结构定义类
    agent.py:           DQN算法智能体实现类
    executer.py:        DQN算法主逻辑执行入口
    selected_data.pk    原始数据的序列化文件
    gen_all_feature_enthalpy.py     原子特征数据与焓变数据序列化文件生成逻辑
    data_feature_enthalpy.pk        原子特征数据与焓变数据序列化文件（由selected_data.pk处理得到）

依赖库:
    numpy, matplotlib, torch, sklearn

    To install a python package, suppose <numpy>, run in terminal:
        pip install numpy
    for more info, contact <xianyeuhui@stu.xjtu.edu.cn>

程序执行：
    在本目录中执行:
        python executer.py