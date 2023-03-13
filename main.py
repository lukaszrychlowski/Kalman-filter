import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, init_x: float, init_v: float, acc_variance: float) -> None:
        self._x = np.array([init_x, init_v])  #state mean grv
        self._acc_variance = acc_variance
        self._P = np.eye(2)                   #covariance of state grv

    def predict(self, dt: float) -> None:
        #x = F * x
        #P = F P Ft + G Gt a
        F = np.array([[1, dt], [0,1]])
        new_x = F.dot(self._x)
        G = np.array([0.5 * dt**2, dt]).reshape((2,1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._acc_variance

        self._P = new_P
        self._x = new_x

    @property
    def position(self) -> float:
        return self._x[0]
    
    @property
    def velocity(self) -> float:
        return self._x[1]

    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def  mean(self) -> np.array:
        return self._x
    

kf = KalmanFilter(init_x=0.2, init_v=0.1, acc_variance=0.2)

