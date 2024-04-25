from __future__ import print_function, division
from tensorboardX import SummaryWriter
import random
import sys
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np




#define the Enviroment
class Enviroment(object):
    def __init__(self, x, y):
        self.train_X = x
        self.train_Y = y
        self.current_index = self._sample_index()
        self.action_space = 129
        self.y_low = min(y)
        self.y_high = max(y)
        self.cal = 0
        self.error = 0
        self.previous_error = 0

    def reset(self):
        obs, _ = self.step(2 ** 10)
        return obs
    #Actions performed at each step
    def step(self, action):
        if action == 2 ** 10:
            _c_index = self.current_index
            # self.current_index = self._sample_index()
            return (self.train_X[_c_index], 0)
        Ty = self.train_Y[self.current_index]
        r, d = self.reward(action, Ty)
        self.current_index = self._sample_index()
        return self.train_X[self.current_index], r, Ty, d

    #define the reward
    def reward(self, action, Ty):
        # #c = self.train_Y[self.current_index]
        # # print(c)
        self.error = np.abs(action - Ty)

        reward = -np.power(self.error, 0.5)
        #
        self.cal += 1
        # if  self.error <= 0.001 or self.cal==200:  # or 0.01 ,self.error <= 0.015 or
        if self.cal == 200:  # or 0.01 ,self.error <= 0.015 or
            self.cal = 0
            done = False
        else:
            done = False
        # print(c)
        # return 1 if c == action else -1
        return reward, done

    def sample_actions(self):
        return random.randint(self.y_low, self.y_high)

    def _sample_index(self):
        return random.randint(0, len(self.train_Y) - 1)




class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, capacity):
        self.storage = []
        self.max_size =capacity
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def save_replay_buffer(self, filepath):
        with open(filepath, 'wb') as f:
            torch.save(torch.tensor(self.storage), f)

    def load_replay_buffer(self, filepath):
        with open(filepath, 'rb') as f:
            self.storage = torch.load(f).tolist()

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d, ty = [], [], [], [], [], []

        for i in ind:
            X, Y, U, R, D, TY = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            ty.append(np.array(TY, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), np.array(
            ty).reshape(-1, 1)



# the base of the Transformer
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V,d_k):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9)                            # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel,d_k,d_v,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.dmodel = dmodel
        self.n_heads= n_heads
        self.d_k= d_k
        self.d_v= d_v
        self.W_Q = nn.Linear(dmodel, int(dmodel / n_heads) * n_heads, bias=False)
        self.W_K = nn.Linear(dmodel, int(dmodel / n_heads) * n_heads, bias=False)
        self.W_V = nn.Linear(dmodel, int(dmodel / n_heads) * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, dmodel, bias=False)

    def forward(self, input_Q, input_K, input_V):  # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, self.d_k)  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.dmodel).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, dmodel, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.dmodel = dmodel
        self.fc = nn.Sequential(
            nn.Linear(dmodel, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, dmodel, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.dmodel).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, dmodel,d_k,d_v,n_heads,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(dmodel,d_k,d_v,n_heads)  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(dmodel,d_ff)  # 前馈神经网络

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, src_len, d_model]

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs
                                               # enc_outputs: [batch_size, src_len, d_model],
                                               )  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


# Transformer Encoder with 1D Convolution
class Actor(nn.Module):

    def __init__(self, d_model, window_size,n_layers,d_k,d_v,n_heads,d_ff):

        super().__init__()
        self.seq_length=window_size
        self.n_layers=n_layers
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 4), 0),
            nn.Conv1d(1, 24, 9, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((3, 3), 0),
            nn.Conv1d(24, 48, 7, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(48, d_model, 5, stride=1),
            nn.ReLU(True)
        )
        self.layers = nn.ModuleList([EncoderLayer(d_model,d_k,d_v,n_heads,d_ff) for _ in range(n_layers)])

        self.dense = nn.Sequential(
            nn.Linear(d_model * self.seq_length, 6 * self.seq_length),
            nn.ReLU(True),
            nn.Linear(6 * self.seq_length, 1),

        )

    def forward(self, src):
        x1 = self.conv(src)

        enc_outputs = x1.view(-1, self.seq_length, self.d_model)  # enc_outputs: [batch_size, src_len, d_model]
        for layer in self.layers:

            enc_outputs, attn = layer(enc_outputs)  # enc_outputs :   [batch_size, src_len, d_model],
        x = self.dense(enc_outputs.view(-1, self.d_model * self.seq_length))


        return x.view(-1, 1)


# the structure of the Critic
class Critic(nn.Module):
    def __init__(self,window_size,state_dim,action_dim):
        super(Critic, self).__init__()

        self.l = nn.Linear(window_size, 1)
        self.s1 = nn.Linear(state_dim + action_dim, 1024)
        self.s2 = nn.Linear(1024, 1024)
        self.s3 = nn.Linear(1024, 780)
        self.s4 = nn.Linear(780, 1)

    def forward(self, enc_inputs, u):  # enc_inputs: [batch_size, src_len]
        enc_inputs = enc_inputs.squeeze(1)
        enc_inputs = torch.cat([enc_inputs, u], 1)
        x = F.relu(self.s1(enc_inputs))
        x = F.relu(self.s2(x))
        x = F.relu(self.s3(x))
        x = self.s4(x)

        return x.view(-1, 1)




device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DDPG(object):
    def __init__(self, model, window_size,layer,action_dim,  directory,batch_size,tau,gamma, policy_frequency,log_interval,d_k,d_v,n_heads,d_ff,capacity):

        self.actor = Actor(model, window_size,layer,d_k,d_v,n_heads,d_ff).to(device)
        self.actor_target = Actor(model, window_size,layer,d_k,d_v,n_heads,d_ff).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic( window_size,window_size,action_dim).to(device)
        self.critic_target = Critic(window_size,window_size,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = Replay_buffer(capacity)
        self.writer = SummaryWriter(directory)
        total_params = sum(p.numel() for p in self.critic.parameters())
        print(f"self.criticNumber of parameters: {total_params}")

        actortotal_params = sum(p.numel() for p in self.actor.parameters())
        print(f"self.actorcNumber of parameters: {actortotal_params}")

        # 计算参数总数
        total_params = sum(p.numel() for p in self.actor.parameters())
        print("Total parameters:", total_params)

        # 遍历并打印每个子模块的参数量
        for name, module in self.actor.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{name} - Parameters: {num_params}")
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.state_dim=window_size
        self.gamma=gamma
        self.log_interval=log_interval
        self.policy_frequency=policy_frequency
        self.batch_size,self.tau=batch_size, tau
        self.d_k,self.d_v=d_k, d_v

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(-1, 1,   self.state_dim)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load_actor(self, path_state_dict):
        self.actor_target = torch.load(path_state_dict)
        self.actor_target.to(device)

    # sampling from the environment and the forward propagation of our network
    def update(self,i):


            x, y, u, r, d, ty = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)


            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward.view(-1, 1) + (done.view(-1, 1) * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            print('critic_loss', critic_loss)
            sys.stdout.flush()
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss

            if i % self.policy_frequency == 0:
                actor_loss = -self.critic(state, self.actor(state)).mean()
                # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                print('actor_loss', actor_loss)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if  self.num_actor_update_iteration % self.log_interval == 0 and i % self.policy_frequency == 0:
                self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                self.writer.add_scalar('Reward/reward', reward.sum() / len(reward),
                                       global_step=self.num_critic_update_iteration)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):

        actor_path_state_dict = "./" + "_actor.pt"
        critic_path_state_dict = "./" + "_critic.pt"


        torch.save(self.actor, actor_path_state_dict)
        torch.save(self.critic, critic_path_state_dict)

        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):  # If you have Convolutional layers, also initialize them
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def initialize(self, layer):
        # Xavier_uniform will be applied to W_{ih}, Orthogonal will be applied to W_{hh}, to be consistent with Keras and Tensorflow
        if isinstance(layer, nn.GRU):
            torch.nn.init.xavier_uniform_(layer.weight_ih_l0.data)
            torch.nn.init.orthogonal_(layer.weight_hh_l0.data)
            torch.nn.init.constant_(layer.bias_ih_l0.data, val=0.0)
            torch.nn.init.constant_(layer.bias_hh_l0.data, val=0.0)
        # Xavier_uniform will be applied to conv1d and dense layer, to be consistent with Keras and Tensorflow
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight.data)
            torch.nn.init.constant_(layer.bias.data, val=0.0)

    def load(self):

        actor_path_state_dict = "./" + "_actor.pt"
        critic_path_state_dict = "./" + "_critic.pt"

        self.actor = torch.load(actor_path_state_dict)
        self.critic = torch.load(critic_path_state_dict)

        print("====================================")
        print("model has been loaded...")
        print("====================================")

