# This is diffusion.py from Janner et al.'s repo: 
# https://github.com/jannerm/diffuser/blob/main/diffuser/models/diffusion.py
# We modify it to incorporate the changes specified in Ajay et al.'s paper:
# 1 - incorporating inverse dynamics
# 2 - use classifier-free guidance with low-temperature sampling

from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, returns, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, returns=returns, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    # values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

class GaussianInvDynDiffusion(nn.Module):
    # TODO: Take in the same arguments as in Ajay et al.'s repo
    # TODO: pass the returns everywhere
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='state_l2', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=True,
        condition_guidance_w=0.1, ar_inv=False, train_only_inv=False
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        # Set all the instance variables Ajay et al. use (most of these aren't actually necessary at the moment, but we need the training script to run)
        self.hidden_dim = hidden_dim
        self.action_weight = action_weight
        self.loss_discount = loss_discount 
        self.loss_weights = loss_weights 
        self.returns_condition=returns_condition 
        self.condition_guidance_w = condition_guidance_w
        self.ar_inv = ar_inv 
        self.train_only_inv = train_only_inv
        
        # Adding Inverse Dynamics
        # Model takes in two states (s_t, s_{t+1}) and outputs the action 
        # that led from s_t to s_{t+1}
        # From the paper: "We represent the inverse dynamics fϕ with a 2-layered 
        # MLP with 512 hidden units and ReLU activations"
        
        self.inv_dyn_model = nn.Sequential(
            nn.Linear(2*observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :] = 0
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns):
        # Modify sampling for classifier-free guidance: Equation (6) in the original
        # CFG paper (https://openreview.net/pdf?id=qw8AKxfYbI)

        noise = (1 + self.condition_guidance_w) * self.model(x, cond, t, use_dropout=False, returns=returns) - self.condition_guidance_w * self.model(x, cond, t, force_dropout=True, returns=returns)

        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, returns=None):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, 0) # TODO: figure out why this argument is set to zero and not action_dim in the original codebase

        # chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x = sample_fn(self, x, cond, returns, t)
            x = apply_conditioning(x, cond, 0)

            progress.update({'t': i})
            # if return_chain: chain.append(x)

        progress.stamp()

        # x, values = sort_by_values(x, values)
        # if return_chain: chain = torch.stack(chain, dim=1)
        return x

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, returns=None):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        # It looks like this should be changed to be just observation_dim (since we're diffusing only over states)
        # shape = (batch_size, horizon, self.transition_dim)
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns=returns)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, returns, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0)

        # NOTE: this dropping out of the conditioning information is handled by the noise model in temporal.py (this is use_dropout=True)

        # With some probability, drop out the class conditioning 
        x_recon = self.model(x_noisy, cond, t, use_dropout=True, returns=returns) 
        x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns):
        # From printing out from the train.py script, it looks like the shape of x
        # is (batch_dim, num_steps_in_trajectory, action_dim +)
        # TODO: Assert this belief LOL
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        # Need to incorporate the second addend from the L(theta, phi)
        # equation at the bottom of page 6 of Ajay et al.'s paper

        # This will require some reshaping magic, since "the inverse dynamics is trained
        # with individual transitions rather than trajectories", but x seems to be a
        # batch of trajectories

        # Extract (s_t, s_{t+1}) pairs out of the dataset
        states = x[:, :, -self.observation_dim:]
        flattened_states = torch.flatten(states, 0, 1)
        num_states = len(flattened_states)
        concat_states = torch.cat((flattened_states[:(num_states - 1), :], flattened_states[1:, :]), dim=1)

        # Need to change the input to p_losses to be only states, *not* both states+actions
        reverse_diffusion_loss, info = self.p_losses(states, cond, returns, t)

        # Extract the actions out of the dataset
        inv_dyn_target = x[:, :, :self.action_dim].flatten(0, 1)[:(num_states - 1), :]

        # Run the inverse dynamics model, get the MSE loss
        inv_dyn_input = self.inv_dyn_model(concat_states)
        inv_dyn_loss = F.mse_loss(inv_dyn_input, inv_dyn_target)

        # return info because the rest of the code will expect two outputs from this function
        return (reverse_diffusion_loss + inv_dyn_loss)/2, info 

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)