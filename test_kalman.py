from main import KalmanFilter
import unittest 

class TestKalmanFilter(unittest.TestCase):
    def test_can_construct(self):
        x = 0.2
        v = 0.1

        kf = KalmanFilter(init_x=x, init_v=v)
        self.assertAlmostEqual(kf.x[0],x)
        self.assertAlmostEqual(kf.x[1],v)