import numpy as np
from src.channel import load_dataset
from src.precoder import sum_rate
from src.baselines import _random_theta

chs = load_dataset("data/train_channels.npz")
rng = np.random.default_rng(0)
   
   # Calculate random baseline on the first 100 training channels
r = np.array([sum_rate(_random_theta(rng, chs[0]["h_r"].shape[1]), ch) for ch in chs[:100]])
   
print("random mean/SD on TRAIN channels:", r.mean(), r.std())
