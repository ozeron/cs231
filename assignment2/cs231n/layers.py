from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    D, M = w.shape
    N = x.shape[0]
    xreshaped = x.reshape((N, D))
    out = xreshaped.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N, M = dout.shape
    D = w.shape[0]
    
    xr = x.reshape((N, D))
    
    dx = dout.dot(w.T).reshape(x.shape)
    dw = xr.T.dot(dout)
    
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout
    dx[x < 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        xsum = np.sum(x, axis=0)
        mean = xsum / N
        xmu = x - mean
        xmu_sq = xmu ** 2
        xmu_sq_sum = np.sum(xmu_sq, axis=0)
        var = xmu_sq_sum / N
        var_eps = var + eps
        sqrt_var_eps = np.sqrt(var_eps)
        ivar = 1 /sqrt_var_eps
        yhat = ivar * xmu
        gamma_yhat = gamma * yhat
        
        out = gamma_yhat + beta
        cache = (x, xsum, mean, xmu, xmu_sq, xmu_sq_sum, var, var_eps, sqrt_var_eps, ivar, yhat, gamma_yhat, gamma, beta)
        
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        xscaled = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * xscaled + beta
        cache = (x, running_mean, running_var, eps, gamma)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    x, xsum, mean, xmu, xmu_sq, xmu_sq_sum, var, var_eps, sqrt_var_eps, ivar, yhat, gamma_yhat, gamma, beta = cache
    
    N,D = x.shape
    
    dbeta = np.sum(dout, axis=0)
    assert dbeta.shape == beta.shape
    dgamma_yhat = dout
    assert dgamma_yhat.shape == gamma_yhat.shape
    
    dgamma = np.sum(dgamma_yhat * yhat, axis=0)
    assert dgamma.shape == gamma.shape
    dyhat = dgamma_yhat * gamma
    assert dyhat.shape == yhat.shape
    
    divar = np.sum(dyhat * xmu, axis=0)
    assert divar.shape == ivar.shape
    dxmu1 = dyhat * ivar
    assert dxmu1.shape == xmu.shape
    
    dsqrt_var_eps = divar * ( -1. * sqrt_var_eps ** (-2.))
    assert dsqrt_var_eps.shape == sqrt_var_eps.shape
    
    dvar_eps = dsqrt_var_eps * (0.5 * (var_eps ** (-0.5)))
    assert dvar_eps.shape == var_eps.shape
    
    dvar = dvar_eps
    assert dvar.shape == var.shape
    
    dxmu_sq_sum = dvar * 1. / N
    assert dxmu_sq_sum.shape == xmu_sq_sum.shape
    
    dxmu_sq = np.ones((N, D)) * dxmu_sq_sum
    assert dxmu_sq.shape == xmu_sq.shape
    
    dxmu2 = dxmu_sq * 2. * xmu
    assert dxmu2.shape == xmu.shape
    
    dxmu = dxmu1 + dxmu2
    assert dxmu.shape == xmu.shape
    dx1 = dxmu
    assert dx1.shape == x.shape
    dmean = -1 * np.sum(dxmu, axis=0)
    assert dmean.shape == mean.shape
    
    dxsum = dmean * 1. / N
    assert dxsum.shape == xsum.shape
    
    dx2 = dxsum * np.ones((N, D))
    assert dx2.shape == x.shape
    
    dx = dx1 + dx2
    assert dx.shape == x.shape
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < (1-p)) / p # first dropout mask. Notice /p!
        out = x * mask # drop!
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, Cf, HH, WW = w.shape
    assert C == Cf
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    
    # pad only H and W 
    xp = np.pad(x, ((0, 0), (0,0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    Hout = 1 + (H + 2*pad - HH) / stride
    Wout = 1 + (W + 2*pad - WW) / stride
    out = np.zeros((N, F, Hout, Wout))
    
    for r in range(N):
        for i in range(Hout):
            for j in range(Wout):
                # top-left
                tl_y = i*stride
                tl_x = j*stride
                x_r = xp[r, :, tl_y:tl_y+HH, tl_x:tl_x+WW]
                out[r, :, i, j]= np.sum(w * x_r, axis=(1,2,3)) + b
                assert x_r.shape == w[0].shape
             
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################   
    x, w, b, conv_param = cache
    
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    Nx, C, H, W = x.shape
    Fw, Cf, HH, WW = w.shape
    N, F, Hout, Wout = dout.shape
    assert N == Nx
    assert F == Fw
    assert Hout == 1 + (H + 2*pad - HH) / stride
    assert Wout == 1 + (W + 2*pad - WW) / stride
    # padded 
    xp = np.pad(x, ((0, 0), (0,0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    
    dx = np.zeros((N, C, H, W))
    dxp = np.zeros(xp.shape)
    dw = np.zeros((F, C, HH, WW))
    
    for n in range(N):
        for f in range(F):
            for i in range(Hout):
                for j in range(Wout):
                    value = dout[n, f, i, j]
                    tl_y = i*stride
                    tl_x = j*stride
                    xgrads = w[f] * value
                    dxp[n, :, tl_y:tl_y+HH, tl_x:tl_x+WW] += xgrads
                    
                    x_row = xp[n, :, tl_y:tl_y+HH, tl_x:tl_x+WW]
                    wgrads = x_row * value
                    dw[f, :, :, :] += wgrads 
                    
                    
    dx = dxp[:, :, pad:-pad, pad:-pad]
    # calculate bias gradient
    db = np.sum(dout, axis=(0, 2, 3))
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    ph = pool_param.get('pool_height', 1)
    pw = pool_param.get('pool_width', 1)
    stride = pool_param.get('stride', 1)
    
    Hout = (H - ph) / stride + 1
    Wout = (H - ph) / stride + 1
    out = np.zeros((N, C, Hout, Wout))
    
    switches = np.zeros(out.shape, dtype=np.int8)
    #for i in range(Hout):
    #    for j in range(Wout):
    #        tl_y = i*stride
    #        tl_x = j*stride
    #        xrow = x[:, :, tl_y:tl_y+2, tl_x:tl_x+2]
    #        maxWal = xrow.max(axis=(2,3)).reshape((N, C))
    #        out[:, :, i, j]= maxWal
    for n in range(N):
        for c in range(C):
            for i in range(Hout):
                for j in range(Wout):
                    tl_y = i*stride
                    tl_x = j*stride
                    xrow = x[n, c, tl_y:tl_y+2, tl_x:tl_x+2]
                    maxarg = xrow.argmax()
                    x_offset, y_offset = np.unravel_index(maxarg, xrow.shape)
                    yi = tl_y + y_offset
                    xi = tl_x + x_offset
                    
                    switches[n, c, i, j] = maxarg
                    out[n, c, i, j]= x[n, c, yi, xi]
                
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, switches)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param, switches = cache
    N, C, H, W = x.shape
    ph = pool_param.get('pool_height', 1)
    pw = pool_param.get('pool_width', 1)
    stride = pool_param.get('stride', 1)
    
    Hout = (H - ph) / stride + 1
    Wout = (H - ph) / stride + 1
    dx = np.zeros(x.shape)
    for n in range(N):
        for c in range(C):
            for i in range(Hout):
                for j in range(Wout):
                    tl_y = i*stride
                    tl_x = j*stride

                    xi, yi = np.unravel_index(switches[n, c, i, j], (ph, pw))
                    tl_y += yi
                    tl_x += xi
                    dx[n, c, tl_y, tl_x] += dout[n, c, i, j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
