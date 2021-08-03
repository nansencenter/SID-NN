import unittest
import numpy as np

from dti_util import _get_col, iter_shuffled, iter_shuffled_old, get_score_importances

class Test_get_col(unittest.TestCase):
    def test_get_col(self):
        M = np.random.rand(10,10,20,10)
        idx = _get_col(1,2,M.ndim)
        result = np.all(M[idx]==M[:,:,1,:])
        self.assertTrue(result)
        
class Test_iter_shuffled_2d(unittest.TestCase):
    def setUp(self):
        self.M = np.random.rand(10,3)
        G1 = iter_shuffled_old(self.M, random_state=1)
        self.L1 = [i.copy() for i in G1]
        G2 = iter_shuffled(self.M, random_state=1, icol=1)
        self.L2 = [i.copy() for i in G2]
    def test_dims(self):
        self.assertEqual(len(self.L2),3)
    def test_shuffle0(self):
        result = np.all(self.L2[1][:,0]==self.L2[2][:,0])
        self.assertTrue(result)
    def test_shuffle1(self):
        result = np.all(self.L2[0][:,1]==self.L2[2][:,1])
        self.assertTrue(result)
    def test_shuffle2(self):
        result = np.all(self.L2[0][:,2]==self.L2[1][:,2])
        self.assertTrue(result)
    def test_equal(self):
        result = np.all((self.L1[0]==self.L2[0]) & (self.L1[1]==self.L2[1]) & (self.L1[2]==self.L2[2]))
        self.assertTrue(result)
        
class Test_iter_shuffled_nd(unittest.TestCase):
    def setUp(self):
        self.M = np.random.rand(500,10,10,3)
        G2 = iter_shuffled(self.M, random_state=1, icol=-1)
        self.L2 = [i.copy() for i in G2]
    def test_dims(self):
        self.assertEqual(len(self.L2),3)
    def test_shuffle0(self):
        result = np.all(self.L2[1][...,0]==self.L2[2][...,0])
        self.assertTrue(result)
    def test_shuffle1(self):
        result = np.all(self.L2[0][...,1]==self.L2[2][...,1])
        self.assertTrue(result)
    def test_shuffle2(self):
        result = np.all(self.L2[0][...,2]==self.L2[1][...,2])
        self.assertTrue(result)
    def test_shuffle0(self):
        result = np.any(self.L2[0][...,0]==self.L2[1][...,0])
        #print(self.L2[0][:,:2,:2,0]==self.L2[1][:,:2,:2,0])
        self.assertFalse(result)
        

def _score(X,y):
    return np.mean(np.square(X[...,0]-y))

class Test_get_score_importances(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(500,10,10,3)
        self.y = self.X[...,0]
        self.base_score, self.score_decreases = get_score_importances(_score, 
                                                    self.X, 
                                                    self.y, 
                                                    n_iter=3
                                                   )
        self.meanscore = np.mean(self.score_decreases, axis=0)
    def test_base_score(self):
        self.assertEqual(self.base_score,0)
    def test_score_decreases_0(self):
        self.assertTrue(self.meanscore[0]<0)
    def test_score_decreases_1(self):
        self.assertTrue(self.meanscore[1]==0)
    def test_score_decreases_2(self):
        self.assertTrue(self.meanscore[2]==0)
        
class Test_get_score_importances_pre(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(500,10,10,3)
        self.y = self.X[...,0]
        self.base_score, self.score_decreases = get_score_importances(_score, 
                                                    self.X, 
                                                    self.y, 
                                                    n_iter=3,
                                                    pre_shuffle=True
                                                   )
        self.meanscore = np.mean(self.score_decreases, axis=0)
    def test_base_score(self):
        self.assertEqual(self.base_score,0)
    def test_score_decreases_0(self):
        self.assertTrue(self.meanscore[0]<0)
    def test_score_decreases_1(self):
        self.assertTrue(self.meanscore[1]==0)
    def test_score_decreases_2(self):
        self.assertTrue(self.meanscore[2]==0)