import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis
data = pd.read_csv("data/1.csv")
data = pd.read_csv("data/test.csv")
print("Successfully install the libraries and read the dataset")