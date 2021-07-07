import numpy as np
from scipy.stats   import rice, foldnorm

class RiceWoolfson_from_parent:
    def __init__(self, parent_R=1, rDW=0, Sigma=1, centric=False):
        """
        This distribution is just a wrapper around the Rice & Woolfson distributions that formulates it in terms of 
        the (DAG) parent structure factor amplitude, the double-Wilson parameter, and Sigma, and uses
        Rice for acentric and Woolfson for centric reflections. Defaults are set such that the function acts as 
        the Wilson distribution. Pirated from Kevin's version in Careless.
        
        Parameters
        ----------
        parent_R : array (float)
                   amplitude of the parent structure factor (default: 1)
        rDW :      float
                   double-Wilson parameter relating the parent and child structure factors (default: 0)
        Sigma :    float (or array of floats with same shape as parent_R)
                   scale parameter (default: 1)
        centric :  array (float or bool)
                   Array of the same shape as parent_R that indicates whether the parent reflections are 
                   centric (1. or True) or acentric (0. or False)
        """
        self._parent_R  = parent_R
        self._rDW       = rDW
        self._Sigma     = Sigma
        self._centric   = np.array(centric, dtype=np.bool)
        self._cond_mean = self._parent_R * self._rDW
        self._cond_varA = 0.5*self._Sigma*(1-self._rDW**2) # acentric
        self._cond_varC =     self._Sigma*(1-self._rDW**2) # centric
        self._cond_sigA = np.sqrt(self._cond_varA)
        self._cond_sigC = np.sqrt(self._cond_varC)
        self._rice_b    = self._cond_mean/self._cond_sigA
        self._foldnorm_c= self._cond_mean/self._cond_sigC
                
        self._woolfson  = foldnorm(c=self._foldnorm_c, scale=self._cond_sigC) 
        self._rice      = rice(    b=self._rice_b,     scale=self._cond_sigA)
        self.eps        = np.finfo(np.float32).eps

    def mean(self):
        return np.where(self._centric, self._woolfson.mean(), self._rice.mean())

    def variance(self):
        return np.where(self._centric, self._woolfson.var(), self._rice.var())

    def stddev(self):
        return np.where(self._centric, self._woolfson.std(), self._rice.std())

    def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
        return np.where(self._centric, self._woolfson.rvs(size=sample_shape, random_state=seed)+self.eps, \
                                       self._rice.rvs(    size=sample_shape, random_state=seed))

    def log_prob(self, x):
        return np.where(self._centric, self._woolfson.logpdf(x), self._rice.logpdf(x))

    def prob(self, x):
        return np.where(self._centric, self._woolfson.pdf(x), self._rice.pdf(x))


class RiceWoolfson_for_DAG:
    def __init__(self, list_of_nodes, list_of_edges, root=0, list_of_rDW=None, root_R=None, \
                 root_rDW=None, Sigma=1, centric=False):
        """
        This is the joint probability distribution for a DAG (if no structure factor amplitudes for the root, root_R, are given),
        or the probability distribution conditional on the root_R with dependence described by root_rDW.
        The class relies on RiceWoolfson_from_parent(), which in turn wraps around the Rice & Woolfson distributions. 
        Defaults are set such that the function acts as the multivariate Wilson distribution if list_of_rdw=None.
        Apologies for the mixed tree/family metaphors! Precise language tbd.
        
        Parameters
        ----------
        list_of_nodes : (list of integers)
                   list of the indices of the nodes of the DAG
        list_of_edges : (list of two-tuples)
                   list of (parent, child) node pairs
        root : (int)
                   index of the node at the base of the DAG (default: 0)
        list_of_rDW : list of floats (len(list_of_edges)), or list of arrays of floats
                   list of the double-wilson parameters, rDW, for each edge in the order that matches list_of_edges
        root_R : (float or array of floats)
                   if not None, an array of structure factor amplitudes for the root of the tree (default: None)
        root_rDW : (float or array of floats)
                   double-Wilson parameter for the dependence of child-of-root structure factors on the root.
                   if root_rDW = None (default), this defaults to the values in list_of_rDW
        Sigma :    float (or array of floats with same shape as parent_R)
                   scale parameter (default: 1)
        centric :  array (float or bool)
                   Array of the same shape as parent_R that indicates whether the parent reflections are 
                   centric (1. or True) or acentric (0. or False)
        """
        self._list_of_nodes = list_of_nodes
        self._n_nodes       = len(list_of_nodes)
        self._list_of_edges = list_of_edges
        self._n_edges       = len(list_of_edges)
        self._root          = root
        self._list_of_rDW   = list_of_rDW
        self._root_R        = root_R
        self._root_rDW      = root_rDW
        self._Sigma         = Sigma
        self._centric       = np.array(centric, dtype=np.bool)
        if root_rDW is not None:
            # when we have empirical rDW per reflection, we'll override the value specified in list_of_rDW[root]
            # if I recall correctly, lists can contain multiple data types
            self._list_of_rDW[self._root] = self._root_rDW
        
    def log_prob(self,x):
        """
        x is an array with an arbitrary number of rows and n_nodes columns.
        note that if root_R is provided to the object, x[:,root] will be disregarded.
        """
        assert x.shape[1] == self._n_nodes
        self._x_nrows = x.shape[0]
        # we should also verify that root_R is a scalar or a numpy array with the same shape[1] as x.
        
        if self._root_R is not None:
            x[:,0] = self._root_R
            print("Overwrote the 0th column with the reference.")
                
        self._log_prob_array = np.zeros((self._x_nrows, self._n_nodes))
        if self._root_R is None:
            # we sample from the Wilson distribution
            self._log_prob_array[:,self._root] = RiceWoolfson_from_parent(parent_R=1, rDW=0, Sigma=self._Sigma, \
                                                              centric=self._centric).log_prob(x[:,self._root])
        # now we follow the tree    
        for e in range(self._n_edges):
            self._log_prob_array[:,self._list_of_edges[e][1]] = \
                               RiceWoolfson_from_parent(parent_R=x[:,self._list_of_edges[e][0]],\
                                                        rDW     =self._list_of_rDW[e],          \
                                                        Sigma   =self._Sigma,                   \
                                                        centric =self._centric                  \
                                                       ).log_prob(x[:,self._list_of_edges[e][1]])
        return np.sum(self._log_prob_array,axis=1)

    def prob(self,x):
        """
        x is an array with an arbitrary number of rows and n_nodes columns.
        note that if root_R is provided to the object, x[:,root] will be disregarded.
        """
        assert x.shape[1] == self._n_nodes
        self._x_nrows = x.shape[0]
        # we should also verify that root_R is a scalar or a numpy array with the same shape[1] as x.
        
        if self._root_R is not None:
            x[:,0] = self._root_R
            print("Overwrote the 0th column with the reference.")
                
        self._prob_array = np.ones((self._x_nrows, self._n_nodes))
        if self._root_R is None:
            # we sample from the Wilson distribution
            self._prob_array[:,self._root] = RiceWoolfson_from_parent(parent_R=1, rDW=0, Sigma=self._Sigma, \
                                                              centric=self._centric).prob(x[:,self._root])
        # now we follow the tree    
        for e in range(self._n_edges):
            self._prob_array[:,self._list_of_edges[e][1]] = \
                               RiceWoolfson_from_parent(parent_R=x[:,self._list_of_edges[e][0]],\
                                                        rDW     =self._list_of_rDW[e],          \
                                                        Sigma   =self._Sigma,                   \
                                                        centric =self._centric                  \
                                                       ).prob(x[:,self._list_of_edges[e][1]])
        return np.prod(self._prob_array,axis=1)

    
    def sample(self, n_samples, seed=None, name='sample', **kwargs):
        """
        Returns an array of size (n_nodes, n_samples,root_R.shape[0]) if root_R is specified as an array
        and (n_nodes, n_samples) otherwise.
        """
        assert isinstance(n_samples, int)
        if self._root_R is not None and not np.isscalar(self._root_R):
            self._n_refl  = self._root_R.shape[0]
        else:
            self._n_refl  = 1
        self._samples_array = np.zeros((self._n_nodes, self._n_refl, n_samples))
                       
        # first assign values for the root, as given, or as sampled.
        if self._root_R is None:
            # we will sample from the Wilson distribution
            print("Sampling the root node from the Wilson distribution...")
            self._samples_array[self._root,0,:] = \
                                    RiceWoolfson_from_parent(parent_R=1, rDW=0, Sigma=self._Sigma, \
                                                            centric=self._centric).sample(sample_shape=(n_samples,),\
                                                                                          seed=None, name='sample', **kwargs)
        else:
            self._samples_array[self._root,:,:] = np.broadcast_to(self._root_R.reshape(self._n_refl,-1), \
                                                                  (self._n_refl, n_samples))
        # now we follow the tree    
        for e in range(self._n_edges):
            for f in range(self._n_refl): # Just to keep things simpler for now
                self._samples_array[self._list_of_edges[e][1],f,:] = \
                           RiceWoolfson_from_parent(parent_R=self._samples_array[self._list_of_edges[e][0],f,:].flatten(),\
                                                    rDW     =self._list_of_rDW[e],          \
                                                    Sigma   =self._Sigma,                   \
                                                    centric =self._centric                  \
                                                   ).sample(sample_shape=(n_samples,), seed=None, name='sample', **kwargs)
        return self._samples_array

