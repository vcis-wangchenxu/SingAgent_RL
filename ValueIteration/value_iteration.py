import numpy as np 
import sys, os
curr_path = os.path.abspath('')
print(curr_path)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
from envs.simple_grid import DrunkenWalkEnv

def all_seed(env, seed = 1):
    import numpy as np
    import random 
    import os
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

env = DrunkenWalkEnv(map_name="theAlley")
all_seed(env, seed=1)

def value_iteration(env, theta=0.005, discount_factor=0.9):    # 一个很小的正数，用于判断收敛性。如果两次迭代之间的Q值变化小于theta，则认为已经收敛。
    Q = np.zeros((env.nS, env.nA))    # initialize a Q table
    count = 0    # 作为计数器，迭代一百次以后即使不收敛也停止迭代
    while True:
        delta = 0.0    # 用于与theta比较，判断是否收敛
        Q_tmp = np.zeros((env.nS, env.nA))
        for state in range(env.nS):
            for action in range(env.nA):
                accum = 0.0
                reward_total = 0.0
                '''
                对于每一个状态-动作对，计算其预期的回报和转移。
                这个预期的回报是当前动作能够获得的即时奖励加上未来所有可能的状态-动作对能够获得的回报的折扣和。
                这个转移是当前状态-动作对的转移概率乘以下一个状态-动作对的最大Q值。
                '''
                for prob, next_state, reward, done in env.P[state][action]:    # prob是转移概率
                    accum += prob * np.max(Q[next_state, :])    # accum += 转移概率*max_Q(s')
                    reward_total += prob * reward               # 转移概率*reward
                Q_tmp[state, action] = reward_total + discount_factor * accum
                delta = max(delta, abs(Q_tmp[state, action] - Q[state, action]))
        Q = Q_tmp

        count += 1
        if delta < theta or count > 100:
            break
    return Q


Q = value_iteration(env)
print(Q)

policy = np.zeros([env.nS, env.nA]) # 初始化一个策略表格
for state in range(env.nS):
    best_action = np.argmax(Q[state, :]) #根据价值迭代算法得到的Q表格选择出策略
    policy[state, best_action] = 1

policy = [int(np.argwhere(policy[i]==1)) for i in range(env.nS) ]
print(policy)


num_episode = 1000 # 测试1000次
def test(env,policy):
    
    rewards = []  # 记录所有回合的奖励
    success = []  # 记录该回合是否成功走到终点
    for i_ep in range(num_episode):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合） 这里state=0
        while True:
            action = policy[state]  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        if state==12: # 即走到终点
            success.append(1)
        else:
            success.append(0)
        rewards.append(ep_reward)
    acc_suc = np.array(success).sum()/num_episode
    print("测试的成功率是：", acc_suc)

test(env, policy)