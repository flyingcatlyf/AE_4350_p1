from ddpg_torch import Agent
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from RK import RK4
from Ref import Ref_wave


env_aircraft = RK4(step=0.1) #set environment

ref_wave = Ref_wave(T=3)

agent = Agent(alpha=0.001, beta=0.001, input_dims=8, lay1_dims=64, lay2_dims=64, n_actions=2,
              gamma=0.99, tau=0.01, env=env_aircraft, batch_size=256, max_size=9000000) #set agent


np.random.seed(0)

#initialization
t0 = np.array([0])
Index0 = 0
score0 = -10000
score_history=[]


for i in range(3000):
    done = False
    score = 0
    step  = 0.1
    amplitude = np.random.uniform(-30 * np.pi / 180, 30 * np.pi / 180)

    x0 = np.array([np.random.uniform(-30*np.pi/180,30*np.pi/180),np.random.uniform(-10*np.pi/180,10*np.pi/180),np.random.uniform(-30*np.pi/180,30*np.pi/180), np.random.uniform(-10*np.pi/180,10*np.pi/180)]) #[roll rate, roll angle, side rate, side angle]obs is the initial value of state variable

    #x0 = np.array([np.uniform(-30*np.pi/180,30*np.pi/180),np.random.uniform(low=-10*np.pi/180,high=10*np.pi/180,size=(1)),np.random.uniform(low=-30*np.pi/180,high=30*np.pi/180,size=(1)), np.random.uniform(low=-10*np.pi/180,high=10*np.pi/180,size=(1))]) #[roll rate, roll angle, side rate, side angle]obs is the initial value of state variable
    #x0 = np.array([np.random.normal(-30*np.pi/180,10*np.pi/180),np.random.normal(-10*np.pi/180,5*np.pi/180),np.random.normal(-30*np.pi/180,10*np.pi/180),np.random.normal(-10*np.pi/180,5*np.pi/180)]) #[roll rate, roll angle, side rate, side angle]obs is the initial value of state variable

    k=0
    while k < 300:

        phi0_ref = ref_wave.fun(k,amplitude)
        x0_ref = np.array([phi0_ref,0,0,0])
        e0 = np.subtract(x0, x0_ref)

        u0 = agent.choose_action(np.concatenate((x0, e0))) #计算actor输出

        #环境一步更新
        new_state, done, info, empty = env_aircraft.solver(t0, x0, u0) #计算当前一步状态

        phi1_ref = ref_wave.fun(k,amplitude)
        x1_ref = np.array([phi1_ref,0,0,0])
        e1 = np.subtract(new_state, x1_ref)

        reward = -10 * abs(np.clip(5*e1[0],-1,1)) - 1 * abs(new_state[1]) - 1 * abs(new_state[3]) - 0.01 * abs(u0[0])- 0.01 * abs(u0[1]) #score=-44

        score = score + reward
        t1 = t0 + 10 * step #时间更新

        #数据记录



        agent.remember(np.concatenate((x0, e0)), np.concatenate((new_state, e1)), u0, reward, int(done))  #存储一步状态转移数据


        #神经网络参数更新
        agent.learn() #训练一次
        agent.learn()  # 训练2次

        #状态重置
        k = k + 1
        t0 = t1
        x0 = new_state

    score_history.append(score)

    if score > score0:
        score0 = score
        Index0 = i + 1
        print('save index', i + 1)
    print('episode', i + 1, 'score % .2f' % score,
          '100 game average %.2f' % np.mean(score_history[-100:]), 'maxscore', score0, 'no.', Index0)
    if i==2999:
        agent.save_models()



np.savetxt('memory1_Action_training',agent.memory1.Action,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory1_NewState_training',agent.memory1.Newstate,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory1_Reward_training',agent.memory1.Reward,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory1_State_training',agent.memory1.State,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)

np.savetxt('memory2_Action_training',agent.memory2.Action,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory2_NewState_training',agent.memory2.Newstate,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory2_Reward_training',agent.memory2.Reward,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)
np.savetxt('memory2_State_training',agent.memory2.State,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)


np.savetxt('scorehistory_training',score_history,delimiter=' ',newline='\n',header='',footer='',comments='#',encoding=None)



