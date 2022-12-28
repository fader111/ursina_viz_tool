import torch
from torch import nn

def get_autoencoder(dense_dim, dim_code, stages=20):
    return nn.Sequential( # основной рабочий
            nn.Linear(dense_dim, dense_dim), # Это полносвязный слой
            nn.ELU(),
            nn.Linear(dense_dim, dense_dim),
            nn.ELU(),
            nn.Linear(dense_dim, dim_code),

            nn.Linear(dim_code, dense_dim*stages),
            nn.ELU(),
            nn.Linear(dense_dim*stages, dense_dim*stages),
            nn.ELU(),
            nn.Linear(dense_dim*stages, dense_dim*stages)
            )