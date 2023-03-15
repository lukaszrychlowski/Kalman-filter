import numpy as np
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter




kf = KalmanFilter(init_x=0, init_v=1, acc_variance=0)
dt = 0.1
steps = 1000

means = []
covs = []

for i in range(steps):
    covs.append(kf.cov)
    means.append(kf.mean)
    kf.predict(dt=dt)

plt.subplot(2,1,1)
plt.title('pos')
plt.plot([mean[0] for mean in means], 'black')
plt.plot([mean[0] - 2*np.sqrt(cov[0,0]) for mean, cov in zip(means, covs)], 'r--')
plt.plot([mean[0] + 2*np.sqrt(cov[0,0]) for mean, cov in zip(means, covs)], 'r--')
 
plt.subplot(2,1,2)
plt.title('velocity')
plt.plot([mean[1] for mean in means], 'black')
plt.plot([mean[1] - 2*np.sqrt(cov[1,1]) for mean, cov in zip(means, covs)], 'r--')
plt.plot([mean[1] + 2*np.sqrt(cov[1,1]) for mean, cov in zip(means, covs)], 'r--')
 
plt.show()