from main import KalmanFilter
import unittest 

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