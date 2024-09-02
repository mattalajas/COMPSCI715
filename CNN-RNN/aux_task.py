#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Retrieved from: https://github.com/facebookresearch/habitat-lab

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Categorical
from typing import Type

def subsampled_mean(x, p=0.1):
    return torch.masked_select(x, torch.rand_like(x) < p).mean()

ACTION_EMBEDDING_DIM = 4

class CPCA(nn.Module):
    """ Action-conditional CPC - up to k timestep prediction
        From: https://arxiv.org/abs/1811.06407
    """
    def __init__(self, hid_size, num_steps, sub_rate, loss_fac, device):
        super(CPCA, self).__init__()

        self.hid_size = hid_size
        self.num_steps = num_steps
        self.sub_rate = sub_rate
        self.loss_fac = loss_fac
        self.device = device

        self.classifier = nn.Sequential(
            nn.Linear(2 * hid_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ) # query and perception
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, hid_size)

    def get_loss(self, actions, vision, belief_features, t, n = 1):
        '''
            actions = num of actions + action embeddings
            vision = next frame embeddings 
            belief_features = final rnn feature at current time step
            t = start of sequence
            n = number of environments per batch
        '''

        k = self.num_steps

        belief_features = belief_features.unsqueeze(0)
        positives = vision
        negative_inds = torch.randperm(k * n, device=self.device)
        negatives = torch.gather(
            positives.reshape(k * n, -1),
            dim=0,
            index=negative_inds.view(k * n, 1).expand(k * n, positives.size(-1)),
        ).view(k, n, -1)
        action_embedding = actions.permute(1, 0, 2) # k x n x -1
        # action_padding = torch.zeros(k - 1, n, ACTION_EMBEDDING_DIM, device=self.device)
        # action_padded = torch.cat((action_embedding, action_padding), dim=0) # k x n x -1
        # t x n x -1 x k
        # action_seq = action_padded.unfold(dimension=0, size=k, step=1)
        # action_seq = action_seq.permute(3, 0, 1, 2)
        # action_seq = action_seq.reshape(k, t*n, ACTION_EMBEDDING_DIM)

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_embedding, belief_features)
        # query_all = out_all.reshape(k, t, n, -1).permute(1, 0, 2, 3)

        positive_input = torch.cat([positives, out_all], -1)
        negative_input = torch.cat([negatives, out_all], -1)
        # Targets: predict k steps for each starting timestep
        # positives_padded = torch.cat((positives[1:], torch.zeros(k, n, self.hid_size, device=self.device)), dim=0) # (t+k) x n
        # positives_expanded = positives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        positives_logits = self.classifier(positive_input)
        # negatives_padded = torch.cat((negatives[1:], torch.zeros(k, n, self.hid_size, device=self.device)), dim=0) # (t+k) x n x -1
        # negatives_expanded = negatives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        negatives_logits = self.classifier(negative_input)

        # Masking
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        # valid_modeling_queries = torch.ones(
        #     t + k, k, n, 1, device=self.device, dtype=torch.bool
        # ) # (padded) timestep predicted x prediction distance x env
        # valid_modeling_queries[t - 1:] = False # >= t is past rollout, and t is index t - 1 here
        # for j in range(1, k + 1): # for j-step predictions
        #     valid_modeling_queries[:j - 1, j - 1] = False # first j frames cannot be valid for all envs (rollout doesn't go that early)
        #     for env in range(n):
        #         has_zeros_batch = env_zeros[env]
        #         # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
        #         # z-> z-1 because of modeling_queries being offset by 1
        #         for z in has_zeros_batch:
        #             valid_modeling_queries[z-1: z-1 + j, j - 1, env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        # valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i) for i in range(t)] # pull the appropriate k per timestep
        # valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2) # t x n x 1 x k -> t x k x n x 1

        # positives_masked_logits = torch.masked_select(positives_logits, valid_mask)
        # negatives_masked_logits = torch.masked_select(negatives_logits, valid_mask)
        positive_loss = F.binary_cross_entropy_with_logits(
            positives_logits, torch.ones_like(positives_logits), reduction='none'
        )

        subsampled_positive = subsampled_mean(positive_loss, p=self.sub_rate)
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives_logits, torch.zeros_like(negatives_logits), reduction='none'
        )
        subsampled_negative = subsampled_mean(negative_loss, p=self.sub_rate)

        aux_losses = subsampled_positive + subsampled_negative
        return aux_losses.mean() * self.loss_fac

class CPCA_Weighted(nn.Module):
    """ To compare with combined aux losses. 5 * k<=1, 4 * k<=2, 3 * k<=4, 2 * k <= 8, 1 * k <= 16 (hardcoded)
        Note - this aux loss is an order of magnitude higher than others (intentionally)
    """
    def __init__(self, cfg, aux_cfg, task_cfg, device):
        super().__init__(cfg, aux_cfg, task_cfg, device)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ) # query and perception
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, cfg.hidden_size)

    def get_loss(self, observations, actions, vision, final_belief_state, belief_features, n, t, env_zeros):
        k = 16

        belief_features = belief_features.view(t*n, -1).unsqueeze(0)
        positives = vision
        negative_inds = torch.randperm(t * n, device=self.device)
        negatives = torch.gather(
            positives.view(t * n, -1),
            dim=0,
            index=negative_inds.view(t * n, 1).expand(t * n, positives.size(-1)),
        ).view(t, n, -1)
        action_embedding = self.action_embedder(actions) # t n -1
        action_padding = torch.zeros(k - 1, n, ACTION_EMBEDDING_DIM, device=self.device)
        action_padded = torch.cat((action_embedding, action_padding), dim=0) # (t+k-1) x n x -1
        # t x n x -1 x k
        action_seq = action_padded.unfold(dimension=0, size=k, step=1).permute(3, 0, 1, 2).view(k, t*n, ACTION_EMBEDDING_DIM)

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features)
        query_all = out_all.view(k, t, n, -1).permute(1, 0, 2, 3)

        # Targets: predict k steps for each starting timestep
        positives_padded = torch.cat((positives[1:], torch.zeros(k, n, self.cfg.hidden_size, device=self.device)), dim=0) # (t+k) x n
        positives_expanded = positives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        positives_logits = self.classifier(torch.cat([positives_expanded, query_all], -1))
        negatives_padded = torch.cat((negatives[1:], torch.zeros(k, n, self.cfg.hidden_size, device=self.device)), dim=0) # (t+k) x n x -1
        negatives_expanded = negatives_padded.unfold(dimension=0, size=k, step=1).permute(0, 3, 1, 2) # t x k x n x -1
        negatives_logits = self.classifier(torch.cat([negatives_expanded, query_all], -1))

        # Masking
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + k, k, n, 1, device=self.device, dtype=torch.bool # not uint so we can mask with this
        ) # (padded) timestep predicted x prediction distance x env
        valid_modeling_queries[t - 1:] = False # >= t is past rollout, and t is index t - 1 here
        for j in range(1, k + 1): # for j-step predictions
            valid_modeling_queries[:j - 1, j - 1] = False # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z-1: z-1 + j, j - 1, env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i) for i in range(t)] # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2) # t x n x 1 x k -> t x k x n x 1

        weight_mask = torch.tensor([5, 4, 3, 3, 2, 2, 2, 2,
                                    1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32,
                                    device=self.device) # this should be multiplied on the loss
        # mask over the losses, not the logits
        positive_loss = F.binary_cross_entropy_with_logits(
            positives_logits, torch.ones_like(positives_logits), reduction='none'
        ) # t k n 1 still

        positive_loss = positive_loss.permute(0, 2, 3, 1) * weight_mask # now t n 1 k
        positive_loss = torch.masked_select(positive_loss.permute(0, 3, 1, 2), valid_mask) # tkn1 again
        subsampled_positive = subsampled_mean(positive_loss, p=self.aux_cfg.subsample_rate)
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives_logits, torch.zeros_like(negatives_logits), reduction='none'
        )
        negative_loss = negative_loss.permute(0, 2, 3, 1) * weight_mask
        negative_loss = torch.masked_select(negative_loss.permute(0, 3, 1, 2), valid_mask)

        subsampled_negative = subsampled_mean(negative_loss, p=self.aux_cfg.subsample_rate)

        aux_losses = subsampled_positive + subsampled_negative
        return aux_losses.mean() * self.aux_cfg.loss_factor