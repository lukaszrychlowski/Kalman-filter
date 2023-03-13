import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, init_x: float, init_v: float) -> None:
        self.x = np.array([init_x, init_v])  #state mean grv
        self.P = np.eye(2)                  #covariance of state grv

kf = KalmanFilter(init_x=0.2, init_v=0.1)

