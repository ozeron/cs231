import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train, dimension = X.shape
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    x_cur = X[i]
    y_cur = y[i]
    scores = X[i].dot(W)
    scores_shifted = scores - np.max(scores)
    probabilities = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted))
#     probabilities = softmax(scores)
#     print(y)
    loss += - np.log(probabilities[y_cur])
    for j in range(num_classes):
        if j == y_cur:
          dW[:, j] += (probabilities[y_cur] - 1) * x_cur
        else:
          dW[:, j] += probabilities[j] * x_cur
        
  loss /= num_train  
  loss += reg * np.sum(W**2)
  
    
  dW /= num_train
  dW += reg * 2* W
  ############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss , dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train, dimension = X.shape
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores_shifted = scores - np.max(scores, axis=1).reshape((scores.shape[0], 1))
  probabilities = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=1).reshape((num_train, 1))
  loss = - np.sum(np.log(probabilities[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W**2)
    
    
  probabilities[np.arange(num_train), y] -= 1

  dW = X.T.dot(probabilities)
  
    
  dW /= num_train
  dW += reg * 2* W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW

