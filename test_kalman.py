from KalmanFilter import KalmanFilter
import unittest 
import numpy as np

class TestKalmanFilter(unittest.TestCase):
    def test_can_construct(self):
        x = 0.2
        v = 0.1
        kf = KalmanFilter(init_x=x, init_v=v, acc_variance=0.2)
        self.assertAlmostEqual(kf.position,x)
        self.assertAlmostEqual(kf.velocity,v)

    def test_can_predict(self):
        x = 0.2
        v = 0.1
        acc_var = 0.1
        kf = KalmanFilter(init_x=x, init_v=v, acc_variance=acc_var)
        kf.predict(dt=1)    

    def test_mean_cov_shapes(self):
        x = 0.2
        v = 0.1
        acc_var = 0.1
        kf = KalmanFilter(init_x=x, init_v=v, acc_variance=acc_var)
        kf.predict(dt=1)   

        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2,))

    def test_call_predict_increases_uncertainty(self):
        x = 0.2
        v = 0.1
        acc_var = 0.1
        kf = KalmanFilter(init_x=x, init_v=v, acc_variance=acc_var)

        for i in range(30):
            old_cov = np.linalg.det(kf.cov)
            kf.predict(dt=1)   
            new_cov = np.linalg.det(kf.cov)
            self.assertGreater(new_cov, old_cov)
            print(old_cov, new_cov)
        
        