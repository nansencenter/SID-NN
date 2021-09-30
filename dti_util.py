import numpy as np
from scipy import ndimage


# Default setting for experiment parameter
default = dict(
    smooth_output=None,strides_test=1,
    smooth_drift=None, smooth_sic=None, smooth_sit=None, scale=True, epsi=None, targetname='d',targetfullname='damage')


def clean_ax(ax):
    for a in ax:
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])

def im2tile(X2,y2, dsize=25,strides=1, subd=1):
    nn,ny,nx,nc = X2.shape
    nrows = (ny-dsize*subd)//strides + 1
    ncols = (nx-dsize*subd)//strides +1

    X1 = np.empty((nrows*ncols*nn,dsize,dsize,nc))
    if y2 is None:
        y1 = None
    else:
        y1 = np.zeros((nrows*ncols*nn,y2.shape[-1]))
    k=0
    for im in range(nn):
        for ix in range(ncols):
            for iy in range(nrows):
                X1[k,:,:,:] = X2[im,iy*strides:iy*strides+dsize*subd:subd,ix*strides:ix*strides+dsize*subd:subd,:]
                if not y2 is None:
                    y1[k,:] = y2[im,iy*strides+(dsize*subd)//2,ix*strides+(dsize*subd)//2,:]
                k=k+1
    return X1,y1

def tile2im(X1,y1,strides=1,ny=400,nx=500, subd=1,squeezey=False):
    dsize = X1.shape[1]
    nrows = (ny-dsize*subd)//strides + 1
    ncols = (nx-dsize*subd)//strides +1
    nn = X1.shape[0]//(nrows*ncols)
    nc = X1.shape[-1]
    X2 = np.zeros((nn,ny,nx,nc))
    if y1 is None:
        y2 = None
    else:
        y2 = np.zeros((nn,ny,nx,y1.shape[-1]))
        
    k=0
    for im in range(nn):
        for ix in range(ncols):
            for iy in range(nrows):
                X2[im,iy*strides:iy*strides+dsize*subd:subd,ix*strides:ix*strides+dsize*subd:subd,:]=X1[k,:,:,:]
                if not y1 is None:
                    y2[im,iy*strides+(dsize*subd)//2,ix*strides+(dsize*subd)//2,:]=y1[k,:]
                k=k+1
    if squeezey:
        y2 = y2[:,(dsize*subd)//2::strides,(dsize*subd)//2::strides]
    return X2, y2


def stack_training(X2, y2, mask_in, mask_out, dsize=25, strides=1, subd=1):
    X2[~mask_in] = np.nan
    if not y2 is None:
        y2[~mask_out] =  np.nan
        if y2.ndim == 3:
            y2 = y2[...,np.newaxis]
    X1, y1 = im2tile(X2, y2, dsize=dsize, strides=strides, subd=subd)
    mask_train_in =  np.all(np.isfinite(X1),axis=(1,2,3)) 
    if y2 is None:
        mask_train = mask_train_in
        y = None
    else:
        mask_train_out =  np.all(np.isfinite(y1),axis=(1)) 
        mask_train = mask_train_in & mask_train_out
        y = y1[mask_train,0]
        
    X = X1[mask_train,:]
    
    
    return X, y, mask_train

def unstack_training(X, y, mask_train, ny=400, nx=500, strides=1, subd=1, squeezey=False):
    nn = mask_train.shape[0]
    nc = X.shape[-1]
    dsize = X.shape[1]
    if y.ndim == 1:
        y = y[...,np.newaxis]
    X1 = np.nan * np.ones((nn,dsize,dsize,nc))
    y1 = np.nan * np.ones((nn,y.shape[-1]))
    X1[mask_train] = X
    y1[mask_train] = y
    X2, y2 = tile2im(X1,y1, strides= strides, ny=ny, nx=nx, subd=subd, squeezey=squeezey)
    return X2, y2

def code_dam(dd, epsi=1e-3, vmin=0.):
    cmin = np.log10(epsi)
    cmax = np.log10(1-vmin+epsi)
    code = np.log10(1-dd+epsi)
    return 2*(code-cmin)/(cmax-cmin) - 1

def decode_dam(cdd, epsi=1e-3, vmin=0.):
    cmin = np.log10(epsi)
    cmax = np.log10(1-vmin+epsi)
    code = (1+ cdd)*(cmax-cmin)/2 + cmin
    return 1+epsi - (10**code)

"""
A module for computing feature importances by measuring how score decreases
when a feature is not available. It contains basic building blocks;
there is a full-featured sklearn-compatible implementation
in :class:`~.PermutationImportance`.

A similar method is described in Breiman, "Random Forests", Machine Learning,
45(1), 5-32, 2001 (available online at
https://www.stat.berkeley.edu/%7Ebreiman/randomforest2001.pdf), with an
application to random forests. It is known in literature as
"Mean Decrease Accuracy (MDA)" or "permutation importance".
"""
#from __future__ import absolute_import
from typing import Tuple, List, Callable, Any

import numpy as np
from sklearn.utils import check_random_state


def iter_shuffled_old(X, columns_to_shuffle=None, pre_shuffle=False,
                  random_state=None):
    """
    Return an iterator of X matrices which have one or more columns shuffled.
    After each iteration yielded matrix is mutated inplace, so
    if you want to use multiple of them at the same time, make copies.

    ``columns_to_shuffle`` is a sequence of column numbers to shuffle.
    By default, all columns are shuffled once, i.e. columns_to_shuffle
    is ``range(X.shape[1])``.

    If ``pre_shuffle`` is True, a copy of ``X`` is shuffled once, and then
    result takes shuffled columns from this copy. If it is False,
    columns are shuffled on fly. ``pre_shuffle = True`` can be faster
    if there is a lot of columns, or if columns are used multiple times.
    """
    rng = check_random_state(random_state)

    if columns_to_shuffle is None:
        columns_to_shuffle = range(X.shape[1])

    if pre_shuffle:
        X_shuffled = X.copy()
        rng.shuffle(X_shuffled)

    X_res = X.copy()
    for columns in columns_to_shuffle:
        if pre_shuffle:
            X_res[:, columns] = X_shuffled[:, columns]
        else:
            rng.shuffle(X_res[:, columns])
        yield X_res
        X_res[:, columns] = X[:, columns]

        
def iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=False,
                  random_state=None, icol=-1):
    """
    Return an iterator of X matrices which have one or more columns shuffled.
    After each iteration yielded matrix is mutated inplace, so
    if you want to use multiple of them at the same time, make copies.

    ``columns_to_shuffle`` is a sequence of column numbers to shuffle.
    By default, all columns are shuffled once, i.e. columns_to_shuffle
    is ``range(X.shape[1])``.

    If ``pre_shuffle`` is True, a copy of ``X`` is shuffled once, and then
    result takes shuffled columns from this copy. If it is False,
    columns are shuffled on fly. ``pre_shuffle = True`` can be faster
    if there is a lot of columns, or if columns are used multiple times.
    """
    rng = check_random_state(random_state)

    if columns_to_shuffle is None:
        columns_to_shuffle = range(X.shape[icol])

    if pre_shuffle:
        X_shuffled = X.copy()
        rng.shuffle(X_shuffled)

    X_res = X.copy()
    ndims = X_res.ndim
    for columns in columns_to_shuffle:
        idx = _get_col(columns, axis=icol, ndims=ndims)
        if pre_shuffle:
            X_res[idx] = X_shuffled[idx]
        else:
            rng.shuffle(X_res[idx])
        yield X_res
        X_res[idx] = X[idx]


def get_score_importances(
        score_func,  # type: Callable[[Any, Any], float]
        X,
        y,
        n_iter=5,  # type: int
        columns_to_shuffle=None,
        random_state=None,
        pre_shuffle=False,
        icol=-1, 
        display=False
    ):
    # type: (...) -> Tuple[float, List[np.ndarray]]
    """
    Return ``(base_score, score_decreases)`` tuple with the base score and
    score decreases when a feature is not available.

    ``base_score`` is ``score_func(X, y)``; ``score_decreases``
    is a list of length ``n_iter`` with feature importance arrays
    (each array is of shape ``n_features``); feature importances are computed
    as score decrease when a feature is not available.

    ``n_iter`` iterations of the basic algorithm is done, each iteration
    starting from a different random seed.

    If you just want feature importances, you can take a mean of the result::

        import numpy as np
        from eli5.permutation_importance import get_score_importances

        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis=0)

    """
    rng = check_random_state(random_state)
    base_score = score_func(X, y)
    scores_decreases = []
    for i in range(n_iter):
        if display:
            print (f'Iteration {i+1}/{n_iter}')
        scores_shuffled = _get_scores_shufled(
            score_func, X, y, columns_to_shuffle=columns_to_shuffle,
            random_state=rng, pre_shuffle = pre_shuffle, icol=icol
        )
        scores_decreases.append(-scores_shuffled + base_score)
    return base_score, scores_decreases



def _get_scores_shufled(score_func, X, y, columns_to_shuffle=None,
                        random_state=None, pre_shuffle=False, icol=-1):
    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state, pre_shuffle=pre_shuffle, icol=icol)
    return np.array([score_func(X_shuffled, y) for X_shuffled in Xs])

def _get_col(icol, axis=-1, ndims=2):
    idx = [slice(None) for i in range(ndims)]
    idx[axis] = icol
    return (tuple(idx))
    

def rmse(yval,ypred):
    return np.sqrt(np.mean(np.square(yval - ypred)))

def corr(yval, ypred):
    return np.corrcoef(yval,ypred)[0,1]

def conv2d(inp,size=3,**kwargs):
    mask = np.isnan(inp)
    inp[mask] = np.median(inp[~mask])
    out = ndimage.uniform_filter(inp,size,**kwargs)
    out[mask] = np.nan
    return(out)
    
    