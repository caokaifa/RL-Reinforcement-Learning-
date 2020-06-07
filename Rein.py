import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt



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
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        returns = []
        policy_loss = []
        self.optimizer.zero_grad()
        # for r, prob in self.data[::-1]:
        for r,prob11 in self.data[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        for prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-torch.log(prob) * R)
        loss=0
        for i in range(len(policy_loss)):
            loss=policy_loss[i]+loss
        loss.backward()
        self.optimizer.step()
        self.data = []
        del self.saved_log_probs[:]


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
            pi.put_data((r, prob[a]))
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
    plt.savefig("reinforce")
    plt.show()
    env.close()


if __name__ == '__main__':
    main()