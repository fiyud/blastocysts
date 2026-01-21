import torch
import torch.nn as nn

from typing import List, Dict, Optional, Tuple


class BaseModel(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
    ) -> None:
        """
        Base Model main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(BaseModel, self).__init__()
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.len_trajectory_pred = len_traj_pred
        if self.learn_angle:
            self.num_action_params = 4  # last two dims are the cos and sin of the angle
        else:
            self.num_action_params = 2

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            obs_img (torch.Tensor): batch of observations
            goal_img (torch.Tensor): batch of goals
        Returns:
            dist_pred (torch.Tensor): predicted distance to goal
            action_pred (torch.Tensor): predicted action
        """
        raise NotImplementedError

class NoMaD(nn.Module):
    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        super(NoMaD, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_img"], input_goal_mask=kwargs["input_goal_mask"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output

class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output
    
