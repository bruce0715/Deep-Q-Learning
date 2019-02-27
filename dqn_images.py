##########################################
# Stat232A&CS266A Project 3:
# Solving CartPole with Deep Q-Network （Images Input）
# Author: Shuai Zhu
##########################################

import argparse
import gym
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


parser = argparse.ArgumentParser(description='DQN_AGENT')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
					help='number of epochs to train (default: 300)')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
					help='batch size for training (default: 32)')
parser.add_argument('--memory-size', type=int, default=10000, metavar='M',
					help='memory length (default: 10000)')
parser.add_argument('--max-step', type=int, default=250,
					help='max steps allowed in gym (default: 250)')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQNagent():
	def __init__(self):
		self.model = DQN()
		self.memory = deque(maxlen=10000)
		self.gamma = 0.9
		self.epsilon_start = 1
		self.epsilon_min = 0.05
		self.epsilon_decay = 200

###################################################################
# remember() function
# remember function is for the agent to get "experience". Such experience
# should be storaged in agent's memory. The memory will be used to train
# the network. The training example is the transition: (state, action,
# next_state, reward). There is no return in this function, instead,
# you need to keep pushing transition into agent's memory. For your
# convenience, agent's memory buffer is defined as deque.
###################################################################
	def remember(self, state, action, reward, next_state,done):
		self.memory.append((state,action,reward,next_state,done))
###################################################################
# act() fucntion
# This function is for the agent to act on environment while training.
# You need to integrate epsilon-greedy in it. Please note that as training
# goes on, epsilon should decay but not equal to zero. We recommend to
# use the following decay function:
# epsilon = epsilon_min+(epsilon_start-epsilon_min)*exp(-1*global_step/epsilon_decay)
# act() function should return an action according to epsilon greedy. 
# Action is index of largest Q-value with probability (1-epsilon) and 
# random number in [0,1] with probability epsilon.
###################################################################
	def act(self, state, steps_done):
		epsilon=self.epsilon_min+(self.epsilon_start-self.epsilon_min)*math.exp(-1*steps_done/self.epsilon_decay)
		a=np.random.random()
		if a<=epsilon:
			action=random.randrange(2)
			return action #greedy search action, random(0,1)
		if a>epsilon:
			#print(self.model(Variable(state)))
			pred_ac=torch.max(self.model(Variable(state)),1)[1] #?
			#print(torch.max(self.model(Variable(state)),1))
			pred_ac=pred_ac.data
			pred_ac=pred_ac.numpy()[0]
			#print(pred_ac)
			return pred_ac

###################################################################
# replay() function
# This function performs an one step replay optimization. It first
# samples a batch from agent's memory. Then it feeds the batch into 
# the network. After that, you will need to implement Q-Learning. 
# The target Q-value of Q-Learning is Q(s,a) = r + gamma*max_{a'}Q(s',a'). 
# The loss function is distance between target Q-value and current
# Q-value. We recommend to use F.smooth_l1_loss to define the distance.
# There is no return of act() function.
# Please be noted that parameters in Q(s', a') should not be updated.
# You may use Variable().detach() to detach Q-values of next state 
# from the current graph.
###################################################################
	def replay(self, batch_size):
		mini_batch=random.sample(self.memory,batch_size)
		series=[]
		actions=[]
		rewards=[]
		next_series=[]
		donelist=[]
		
		for s in mini_batch:
			series.append(s[0])
			actions.append(s[1])
			rewards.append(int(s[2]))
			next_series.append(s[3])
			donelist.append(s[4])
    
		series=torch.cat(series)
		actions=np.array(actions)
		rewards=np.array(rewards)
		next_series=torch.cat(next_series)
		donelist=np.array(donelist)
		#print(torch.max(series))
        #my predict, 1-dim FloatTensor Variable
		act_index=Variable(torch.from_numpy(actions).long()).view(-1,1)
		#print(act_index)
		target_f=self.model(Variable(series)).gather(1,act_index)
		#print(self.model(Variable(series)))    
        
        #approximate true value
		rewards=Variable(torch.from_numpy(rewards).float()) #1,-1 #floattensor
		index=rewards!=-1 #this is also a variable
		index=index.float()
        
        #predict future network to approximate
		target_a=torch.max(self.model(Variable(next_series)),dim=1)[0]
		#print(target_a)
		#print(index)
		target_a=torch.mul(target_a,index)#remove predicted reward with those done=true
		target_a=rewards+self.gamma*target_a
		#print(target_a-target_f)
        
        #define loss and bp
		loss=F.smooth_l1_loss(target_f,target_a.detach())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


#################################################################
# Functions 'getCartLocation' and 'getGymScreen' are designed for 
# capturing current renderred image in gym. You can directly take 
# the return of 'getGymScreen' function, which is a resized image
# with size of 3*40*80.
#################################################################

def getCartLocation():
	world_width = env.x_threshold*2
	scale = 600/world_width
	return int(env.state[0]*scale+600/2.0)

def getGymScreen():
	screen = env.render(mode='rgb_array').transpose((2,0,1))
	screen = screen[:, 160:320]
	view_width = 320
	cart_location = getCartLocation()
	if cart_location < view_width//2:
		slice_range = slice(view_width)
	elif cart_location > (600-view_width//2):
		slice_range = slice(-view_width, None)
	else:
		slice_range = slice(cart_location - view_width//2, cart_location+view_width//2)
	screen = screen[:, :, slice_range]
	screen = np.ascontiguousarray(screen, dtype=np.float32)/255
	screen = torch.FloatTensor(screen)
	return resize(screen).unsqueeze(0)

def plot_durations(durations):
    #moving average 30
    i=30
    m_durations=[]
    while i <=len(durations):
        dur_mean=np.mean(durations[i-30:i])
        m_durations.append(dur_mean)
        i+=1
    m_durations=[0]*30+m_durations
    plt.figure(2)
    plt.clf()
    plt.title('Training results')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations)
    plt.plot(m_durations)
    plt.plot()
    plt.show()  



env = gym.make('CartPole-v0').unwrapped
env._max_episode_steps = args.epochs
print('env max steps:{}'.format(env._max_episode_steps))
steps_done = 0
agent = DQNagent()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.model.parameters()), lr=1e-3)
durations = []
	################################################################
	# training loop
	# You need to implement the training loop here. In each epoch, 
	# play the game until trial ends. At each step in one epoch, agent
	# need to remember the transitions in self.memory and perform
	# one step replay optimization. Use the following function to 
	# interact with the environment:
	#   env.step(action)
	# It gives you infomation about next step after taking the action.
	# The return of env.step() is (next_state, reward, done, info). You
	# do not need to use 'info'. 'done=1' means current trial ends.
	# if done equals to 1, please use -1 to substitute the value of reward.
	################################################################
for epoch in range(1, args.epochs+1): #cahnge back
	steps = 0 #游戏计分器
	env.reset()
	last_screen = getGymScreen()
	current_screen = getGymScreen()
	state = current_screen - last_screen 
	#print(state.size())
	#3*40*80 FloatTensor
	for steps in range(args.max_step+1): #最多250分
		steps_done+=1 #用来做eplision search
		action=agent.act(state,steps_done) #用eplision search得到action
        #calculate next_state
		last_screen=current_screen
		_,reward,done,_=env.step(action) #与环境互动
		current_screen=getGymScreen()
		next_state=current_screen-last_screen
# 		print(torch.max(next_state))
# 		print(torch.min(next_state))
		#check if game over
		if steps==args.max_step:
			durations.append(steps)
			print("episode:{}/{},score:{}".format(epoch,args.epochs,steps))
			break
            
		if done:
			durations.append(steps) #how many rewards I got in this epsiode
			reward=-1 #punish for ending game
			agent.remember(state,action,reward,next_state,done) #put into memory
			if steps_done>=args.batch_size:
				agent.replay(args.batch_size)
			print("episode:{}/{},score:{}".format(epoch,args.epochs,steps))
			break

			#put into memory
		agent.remember(state,action,reward,next_state,done)

			#experience replay sample batch from memory to update network
		if steps_done>=args.batch_size:
			agent.replay(args.batch_size)

			#make next_state the current one
		state=next_state



plot_durations(durations)
