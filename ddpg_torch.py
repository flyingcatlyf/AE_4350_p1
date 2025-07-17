import os
import torch.nn as nn
import torch.optim as optim
import torch as T
import torch.nn.functional as F
import numpy as np


class ActionNoise(object):
    def __init__(self, action, sigma=0.015, theta=0.01, dt=0.01, x0=None):
        self.action = action
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0

        self.reset()

    def __call__(self, *args, **kwargs):
        x = self.x_prev + self.theta * (self.action-self.x_prev)*self.dt+self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.action.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.action)


class ReplayBuffer(object):
    def __init__(self, max_size, state_shape, n_actions):
        self.max_size = max_size
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.index = 0

        self.State = np.zeros((self.max_size, self.state_shape))
        self.Newstate = np.zeros((self.max_size, self.state_shape))
        self.Action = np.zeros((self.max_size, self.n_actions))
        self.Reward = np.zeros(self.max_size)

        self.terminal_pool = np.zeros(self.max_size, dtype=np.float32)

    def store_transition(self, state, newstate, action, reward, done):
        index = self.index % self.max_size

        self.State[index] = state
        self.Newstate[index] = newstate
        self.Action[index] = action
        self.Reward[index] = reward
        self.terminal_pool[index] = 1-done
        self.index = self.index + 1

    def sampling(self, batch_size):
        samplesize_max = min(self.index, self.max_size)#samplesize_max is the actual data size in the buffer.
        batch = np.random.choice(samplesize_max, batch_size)

        states = self.State[batch]
        newstates = self.Newstate[batch]
        actions = self.Action[batch]
        rewards = self.Reward[batch]
        terminals = self.terminal_pool[batch]

        return states, newstates, actions, rewards, terminals


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, lay1_dims, lay2_dims, n_actions, name,
                chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.lay1_dims = lay1_dims
        self.lay2_dims = lay2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')
        #print('checkpointfile=',self.checkpoint_file)

        self.lay1 = nn.Linear(self.input_dims, self.lay1_dims)
        self.lay2 = nn.Linear(self.lay1_dims, self.lay2_dims)
        self.lay3 = nn.Linear(self.lay2_dims, self.n_actions)

        self.bn1 = nn.LayerNorm(self.lay1_dims)
        self.bn2 = nn.LayerNorm(self.lay2_dims)

        self.lay11 = nn.Linear(self.n_actions, self.lay2_dims)
        self.lay21 = nn.Linear(self.lay2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda：0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        lay1_out = self.lay1(state)
        lay1_out = self.bn1(lay1_out)
        lay1_out = F.relu(lay1_out)

        lay2_out = self.lay2(lay1_out)
        lay2_out = self.bn2(lay2_out)

        lay11_out = self.lay11(action)
        lay11_out = F.relu(lay11_out)

        sum = T.add(lay2_out, lay11_out)
        lay21_in = F.relu(sum)
        lay21_out = self.lay21(lay21_in)

        return lay21_out

    def save_checkpoint(self):
        print('... saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, lay1_dims, lay2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.lay1_dims = lay1_dims
        self.lay2_dims = lay2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        self.lay1 = nn.Linear(self.input_dims, self.lay1_dims)
        self.lay2 = nn.Linear(self.lay1_dims, self.lay2_dims)
        self.lay3 = nn.Linear(self.lay2_dims, n_actions)

        self.bn1 = nn.LayerNorm(self.lay1_dims)
        self.bn2 = nn.LayerNorm(self.lay2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda：0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        lay1_out = self.lay1(state)
        lay1_out = self.bn1(lay1_out)
        lay1_out = T.tanh(lay1_out)

        lay2_out = self.lay2(lay1_out)
        lay2_out = self.bn2(lay2_out)
        lay2_out = F.relu(lay2_out)

        lay3_out = self.lay3(lay2_out)
        lay3_out = T.tanh(lay3_out)

        return lay3_out

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, alpha, beta, input_dims, lay1_dims, lay2_dims, n_actions, gamma, tau, env, batch_size, max_size):
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.lay1_dims = lay1_dims
        self.lay2_dims = lay2_dims
        self.n_actions = n_actions

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_size = max_size

        self.state_shape = input_dims

        #Structure
        self.noise = ActionNoise(action=np.zeros(self.n_actions))
        self.memory1 = ReplayBuffer(self.max_size, self.state_shape, self.n_actions)
        self.memory2 = ReplayBuffer(self.max_size, self.state_shape, self.n_actions)

        self.critic1 = CriticNetwork(self.beta,self.input_dims,self.lay1_dims,self.lay2_dims,self.n_actions,name='Critic1')
        self.critic2 = CriticNetwork(self.beta,self.input_dims,self.lay1_dims,self.lay2_dims,self.n_actions,name='Critic2')

        self.actor = ActorNetwork(self.alpha, self.input_dims, self.lay1_dims,self.lay2_dims,self.n_actions,name='Actor')

        self.target_critic1 = CriticNetwork(self.beta, self.input_dims, self.lay1_dims, self.lay2_dims, self.n_actions,name='TargetCritic1')
        self.target_critic2 = CriticNetwork(self.beta, self.input_dims, self.lay1_dims, self.lay2_dims, self.n_actions,name='TargetCritic2')

        self.target_actor = ActorNetwork(self.alpha, self.input_dims, self.lay1_dims, self.lay2_dims, self.n_actions, name='TargetActor')

        self.update_network_parameters1(tau=1)
        self.update_network_parameters2(tau=1)


    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)

        action = self.actor(observation).to(self.actor.device) #计算控制器输出
        #action_noise = action
        action_noise = action + T.tensor(self.noise(), dtype=T.float).to(self.actor.device) #控制器加噪声

        self.actor.train()
        return action_noise.cpu().detach().numpy()

    def remember(self, state, new_state, action, reward, done):
        self.memory1.store_transition(state, new_state, action, reward, done)

        #data augmentation
        state2 = -state
        new_state2 = -new_state
        action2 = -action
        reward2 = reward
        done2 = done
        self.memory2.store_transition(state2, new_state2, action2, reward2, done2)

    def learn(self):
        #========================================train critic1========================================================
        if self.memory1.index < self.batch_size:
            return
        states1, newstates1, actions1, rewards1, dones1= self.memory1.sampling(self.batch_size) #1.sampling from replay buffer.

        #print('states=', states)
        #print('newstates=', newstates)
        #print('actions=',actions)
        #print('rewards=', rewards)

        states1 = T.tensor(states1, dtype=T.float).to(self.critic1.device)#2.transfer samples to tensor
        newstates1 = T.tensor(newstates1, dtype=T.float).to(self.critic1.device)
        actions1 = T.tensor(actions1, dtype=T.float).to(self.critic1.device)
        rewards1 = T.tensor(rewards1, dtype=T.float).to(self.critic1.device)#[64]
        rewards1=rewards1.view(self.batch_size, 1) #shape=[64,1]
        dones1 = T.tensor(dones1).to(self.critic1.device)
        dones1 = dones1.view(self.batch_size, 1)
        #print('dones=',dones)


        self.critic1.eval()
        self.target_critic1.eval()
        self.target_actor.eval()

        #forwards
        critic_out1 = self.critic1.forward(states1, actions1)#Q0s
        target_actor_out1 = self.target_actor.forward(newstates1)  # Actions1
        target_critic_out1 = self.target_critic1.forward(newstates1,target_actor_out1)#Q1s


        target1 = [] #an empety list
        for j in range(self.batch_size):
            target1.append(rewards1[j] + self.gamma*target_critic_out1[j]*dones1[j])

        target1 = T.tensor(target1).to(self.critic1.device)
        target1 = target1.view(self.batch_size, 1)


        #train critic1
        self.critic1.train()
        self.critic1.optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_out1, target1) #calculate critic loss.
        #critic_loss = F.cross_entropy(critic_out, target)
        critic_loss.backward() #calculate the gradient.
        self.critic1.optimizer.step() #update the parameters of critic.


        #================================train critic 2==========================================================
        if self.memory2.index < self.batch_size:
            return
        states2, newstates2, actions2, rewards2, dones2 = self.memory2.sampling(self.batch_size)

        states2 = T.tensor(states2, dtype=T.float).to(self.critic2.device)  # 2.transfer samples to tensor
        newstates2 = T.tensor(newstates2, dtype=T.float).to(self.critic2.device)
        actions2 = T.tensor(actions2, dtype=T.float).to(self.critic2.device)
        rewards2 = T.tensor(rewards2, dtype=T.float).to(self.critic2.device)  # [64]
        rewards2 = rewards2.view(self.batch_size, 1)  # shape=[64,1]
        dones2 = T.tensor(dones2).to(self.critic2.device)
        dones2 = dones2.view(self.batch_size, 1)

        self.critic2.eval()
        self.target_critic2.eval()
        self.target_actor.eval()

        #forwards
        critic_out2 = self.critic2.forward(states2, actions2)#Q0s
        target_actor_out2 = self.target_actor.forward(newstates2)  # Actions1
        target_critic_out2 = self.target_critic2.forward(newstates2,target_actor_out2)#Q1s

        target2 = [] #an empety list
        for j in range(self.batch_size):
           target2.append(rewards2[j] + self.gamma*target_critic_out2[j]*dones2[j])

        target2 = T.tensor(target2).to(self.critic2.device)
        target2 = target2.view(self.batch_size, 1)

        # train critic2
        self.critic2.train()
        self.critic2.optimizer.zero_grad()
        critic_loss2 = F.mse_loss(critic_out2, target2)  # calculate critic loss.
        critic_loss2.backward()  # calculate the gradient.
        self.critic2.optimizer.step()  # update the parameters of critic.


        #====================================1st time train actor==================================================
        self.critic1.eval()
        self.actor.optimizer.zero_grad()
        actor_outs1 = self.actor.forward(states1)#Actions0, output of actor network

        self.actor.train()
        actor_loss = -self.critic1.forward(states1, actor_outs1) #calculate actor loss/Q with chosen Actions
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters1()

        #====================================2nd time train actor==================================================
        self.critic2.eval()
        self.actor.optimizer.zero_grad()
        actor_outs2 = self.actor.forward(states2)#Actions0, output of actor network

        self.actor.train()
        actor_loss2 = -self.critic2.forward(states2, actor_outs2) #calculate actor loss/Q with chosen Actions
        actor_loss2 = T.mean(actor_loss2)
        actor_loss2.backward()
        self.actor.optimizer.step()

        self.update_network_parameters2()

    def update_network_parameters1(self, tau=None):#update target_critic/target_actor parameters
        if tau is None:
            tau = self.tau

        critic_params1 = self.critic1.named_parameters()
        actor_params = self.actor.named_parameters()
        target_critic_params1 = self.target_critic1.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        critic_params_dict1 = dict(critic_params1)
        actor_params_dict = dict(actor_params)
        target_critic_params_dict1 = dict(target_critic_params1)
        target_actor_params_dict = dict(target_actor_params)

        for name in critic_params_dict1:
            critic_params_dict1[name] = tau*critic_params_dict1[name].clone() + \
                                    (1-tau)*target_critic_params_dict1[name].clone()

        self.target_critic1.load_state_dict(critic_params_dict1)

        for name in actor_params_dict:
            actor_params_dict[name] = tau*actor_params_dict[name].clone() + \
                                     (1-tau)*target_actor_params_dict[name].clone()

        self.target_actor.load_state_dict(actor_params_dict)

    def update_network_parameters2(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_params2 = self.critic2.named_parameters()
        actor_params = self.actor.named_parameters()
        target_critic_params2 = self.target_critic2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        critic_params_dict2 = dict(critic_params2)
        actor_params_dict = dict(actor_params)
        target_critic_params_dict2 = dict(target_critic_params2)
        target_actor_params_dict = dict(target_actor_params)

        for name in critic_params_dict2:
            critic_params_dict2[name] = tau*critic_params_dict2[name].clone() + \
                                    (1-tau)*target_critic_params_dict2[name].clone()

        self.target_critic2.load_state_dict(critic_params_dict2)

        for name in actor_params_dict:
            actor_params_dict[name] = tau*actor_params_dict[name].clone() + \
                                     (1-tau)*target_actor_params_dict[name].clone()

        self.target_actor.load_state_dict(actor_params_dict)

    def save_models(self):
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()

        self.actor.save_checkpoint()

        self.target_critic1.save_checkpoint()
        self.target_critic2.save_checkpoint()

        self.target_actor.save_checkpoint()

    def load_models(self):
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()

        self.actor.load_checkpoint()

        self.target_critic1.load_checkpoint()
        self.target_critic2.load_checkpoint()

        self.target_actor.load_checkpoint()














#BB = ActorNetwork(alpha=0.01, input_dims=8, lay1_dims=400, lay2_dims=300, n_actions=2, name='actor')
#state = T.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=T.float)
#b = BB.forward(state)
#print(b)


#AA = CriticNetwork(beta=0.01, input_dims=8, lay1_dims=400, lay2_dims=300, n_actions=2, name='critic')
#state = T.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=T.float)
#action = T.tensor([0, 0], dtype=T.float)

#s=AA.lay1_dims
#print(AA.input_dims)
#s=CriticNetwork.lay1(state)


#s = AA.forward(state, action)
#print(s)

# AA.forward(T.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=T.float),T.tensor([1, 1], dtype=T.float))

# AA.lay1(state1)
# AA.forward(state,action1)
# critic_out = CriticNetwork.forward(state1, state1, action1)

# print(critic_out)

# def __int__(self, beta, input_dims, lay1_dims, lay2_dims, n_actions, name,

# state1 = T.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=T.float)

# a=nn.Linear(8,300)                #chkpt_dir='tmp/ddpg'):
# out=a(state1)
# print(out.size())
