# Code retrieved from: https://github.com/higgsfield-ai/higgsfield
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import timm
from torch.distributions import Normal
from CNNRNN import models

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self, in_dim, conv_out, rnn_type, rnn_emb, act_dim, final_out, dropout = 0):
        super(Actor, self).__init__()
        
        # self.conv = models.LeNet(img_size, conv_out, padding, kernel, stride, dropout)
        assert rnn_type.lower() == 'gru' or rnn_type.lower() == 'lstm' 
        
        self.rnn_type = rnn_type
        if self.rnn_type.lower() == 'gru':
            self.rnn = models.actionGRU(in_dim, rnn_emb, act_dim, conv_out, dropout)
        else:
            self.rnn = models.actionLSTM(in_dim, rnn_emb, act_dim, conv_out, dropout)
        
        self.mlp = models.MLP(conv_out, final_out, dropout)
        
    def forward(self, img, actions, h0, c0=0):
        # image_r = self.conv(img)

        # GRU step per image and its associated thumbstick comman
        if self.rnn_type.lower() == 'gru':
            h0 = self.rnn(img, actions, h0)
        else:
            h0, c0 = self.rnn(img, actions, h0, c0)
        
        h0 = F.relu(h0)

        # Final prediction for each frame 
        fin = self.mlp(h0)
        return fin, h0, c0
    
class Critic(nn.Module):
    def __init__(self, in_dim, conv_out, rnn_type, rnn_emb, act_dim, dropout = 0):
        super(Critic, self).__init__()
        
        # self.conv = models.LeNet(img_size, conv_out, padding, kernel, stride, dropout)
        assert rnn_type.lower() == 'gru' or rnn_type.lower() == 'lstm' 
        
        self.rnn_type = rnn_type
        if self.rnn_type.lower() == 'gru':
            self.rnn = models.actionGRU(in_dim, rnn_emb, act_dim, conv_out, dropout)
        else:
            self.rnn = models.actionLSTM(in_dim, rnn_emb, act_dim, conv_out, dropout)
        
        self.mlp = models.MLP(conv_out, 1, dropout)
        
    def forward(self, img, actions, h0, c0=0):
        # image_r = self.conv(img)

        # GRU step per image and its associated thumbstick comman
        if self.rnn_type.lower() == 'gru':
            h0 = self.rnn(img, actions, h0)
        else:
            h0, c0 = self.rnn(img, actions, h0, c0)

        h0 = F.relu(h0)

        # Final prediction for each frame 
        fin = self.mlp(h0)
        return fin, h0, c0

class ActorCritic(nn.Module):
    def __init__(self, num_outputs, conv_out, rnn_type, rnn_emb, \
                 act_dim, final_out, dropout = 0, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = Critic(num_outputs, conv_out, rnn_type, rnn_emb, act_dim, dropout)
        
        # Make actor into CNN with RNN maybe
        self.actor = Actor(num_outputs, conv_out, rnn_type, rnn_emb, act_dim, final_out, dropout)

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, img, actions, h0_c, h0_a, c0_c=0, c0_a=0):
        value, h1_c, c1_c = self.critic(img, actions, h0_c, c0_c)
        mu, h1_a, c1_a = self.actor(img, actions, h0_a, c0_a)

        value = F.relu(value)
        mu = F.relu(mu)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value, h1_c, h1_a, c1_c, c1_a

class ActorCriticConv(nn.Module):
    def __init__(self, num_outputs, img_size, conv_out, rnn_type, rnn_emb, \
                 act_dim, final_out, dropout = 0, std=0.0):
        super(ActorCriticConv, self).__init__()

        self.lenet = models.LeNet(img_size, conv_out, dropout=dropout)
        
        self.critic = Critic(num_outputs, conv_out, rnn_type, rnn_emb, act_dim, dropout)
        
        # Make actor into CNN with RNN maybe
        self.actor = Actor(num_outputs, conv_out, rnn_type, rnn_emb, act_dim, final_out, dropout)

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, read_img, state, actions, h0_c, h0_a, c0_c=0, c0_a=0):
        img = read_img(state[0], state[1])
        img = F.relu(self.lenet(img))

        value, h1_c, c1_c = self.critic(img, actions, h0_c, c0_c)
        mu, h1_a, c1_a = self.actor(img, actions, h0_a, c0_a)

        value = F.relu(value)
        mu = F.relu(mu)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value, h1_c, h1_a, c1_c, c1_a
    
class ActorCriticSingle(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCriticSingle, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = F.sigmoid(self.actor(x))
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size, rnn_type, dropout = 0):
        super(Discriminator, self).__init__()

        self.hlinear = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.clinear = nn.Linear(hidden_size + hidden_size, hidden_size)

        assert rnn_type.lower() == 'gru' or rnn_type.lower() == 'lstm' 
        
        self.rnn_type = rnn_type
        if self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=num_inputs, hidden_size=hidden_size)
        else:
            self.rnn = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size)

        self.drop1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.mul_(0.0)
    
    def forward(self, x, h0_c, h0_a, c0_c = 0, c0_a = 0):
        if self.rnn_type.lower() == 'gru':
            h0 = self.hlinear(torch.cat([h0_c, h0_a], 1))
            h0 = torch.unsqueeze(h0, 0)

            _, h = self.rnn(x, h0)
        else:
            h0 = self.hlinear(torch.cat([h0_c, h0_a], 1))
            c0 = self.clinear(torch.cat([c0_c, c0_a], 1))

            h0 = torch.unsqueeze(h0, 0)
            c0 = torch.unsqueeze(c0, 0)

            _, h = self.rnn(x, (h0, c0))

        h = self.drop1(h)
        h = F.tanh(self.linear1(h))
        out = F.sigmoid(self.linear2(h))
        out = out.view(-1, 1)

        # num_dims = len(out.shape)
        # reduction_dims = tuple(range(1, num_dims))
        # out = torch.mean(out, dim=reduction_dims)

        return out

class DiscriminatorConv(nn.Module):
    def __init__(self, num_inputs, img_size, hidden_size, rnn_type, dropout = 0):
        super(DiscriminatorConv, self).__init__()

        self.lenet = models.LeNet(img_size, hidden_size, dropout=dropout)
        self.hlinear = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.clinear = nn.Linear(hidden_size + hidden_size, hidden_size)

        assert rnn_type.lower() == 'gru' or rnn_type.lower() == 'lstm' 
        
        self.rnn_type = rnn_type
        if self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=num_inputs, hidden_size=hidden_size)
        else:
            self.rnn = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size)

        self.drop1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.mul_(0.0)
    
    def forward(self, img, h0_c, h0_a, c0_c = 0, c0_a = 0):
        x = F.relu(self.lenet(img))

        if self.rnn_type.lower() == 'gru':
            h0 = self.hlinear(torch.cat([h0_c, h0_a], 1))
            h0 = torch.unsqueeze(h0, 0)

            x, _ = self.rnn(x, h0)
        else:
            h0 = self.hlinear(torch.cat([h0_c, h0_a], 1))
            c0 = self.clinear(torch.cat([c0_c, c0_a], 1))

            h0 = torch.unsqueeze(h0, 0)
            c0 = torch.unsqueeze(c0, 0)

            x, _= self.rnn(x, h0, c0)

        x = self.drop1(x)
        x = F.tanh(self.linear1(x))
        x = F.sigmoid(self.linear2(x))

        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        out = torch.mean(out, dim=reduction_dims)

        return out
    
class DiscriminatorSingle(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(DiscriminatorSingle, self).__init__()
        
        self.linear1   = nn.Linear(num_inputs, hidden_size)
        self.linear2   = nn.Linear(hidden_size, hidden_size)
        self.linear3   = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)
    
    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        prob = F.sigmoid(self.linear3(x))
        return prob

class TrainedResNet(nn.Module):
    def __init__(self, final_out, freeze = True):
        super(TrainedResNet, self).__init__()
        # Load a pre-trained ResNet-50 model from timm
        self.resnet_model = timm.create_model('resnet50', pretrained=True)

        # Freeze ResNet weights (optional)
        for param in self.resnet_model.parameters():
            param.requires_grad = freeze # True or False

        hid_size = self.resnet_model.fc.out_features
        # Modify ResNet output
        
        self.fc1 = nn.Linear(hid_size, hid_size//2)
        self.fc2 = nn.Linear(hid_size//2, final_out)
        
    def forward(self, x):
        res = F.sigmoid(self.resnet_model(x))

        out = F.relu(self.fc1(res))
        out = self.fc2(out)
        return out