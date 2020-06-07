import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np


# Hyperparameters
learning_rate = 0.0002
gamma = 0.98


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.saved_log_probs = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch


    def train_net(self):

        re=[]
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        # for i in range(len(self.saved_log_probs)):
        #     re.append([self.saved_log_probs[i]])
        # pi_a=torch.tensor(re)

        pi = self.forward(s)
        pi_a = pi.gather(1, a)

        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.saved_log_probs=[]



def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    plreward=[]
    plepoch=[]
    for n_epi in range(5000):
        s = env.reset()
        done = False

        while not done:  # CartPole-v1 forced to terminates at 500 step.
            env.render()
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            pi.saved_log_probs.append(prob[a])
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((s, a, r, s_prime, done))
            s = s_prime
            score += r

        pi.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
            plepoch.append(n_epi)
            plreward.append(score / print_interval)
            score = 0.0



    plt.plot(plepoch,plreward)
    plt.title("reward")
    plt.savefig("AC")
    plt.show()
    env.close()


if __name__ == '__main__':
    main()