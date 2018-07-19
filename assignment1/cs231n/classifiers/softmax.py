
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  dS = np.zeros((num_train, num_class))
  for i, x in enumerate(X):
        score = np.dot(x, W)
        score = np.exp(score)
        loss += -1 * np.log(score[y[i]] / np.sum(score))
        sum_score = np.sum(score)
        dS[i] = score / sum_score 
        dS[i][y[i]] -= 1
  dW = X.T.dot(dS) / num_train + 2 * reg * W
  loss = loss / num_train + reg * np.sum(W * W)
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train, num_class = X.shape[0], W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  score = np.dot(X, W)
  score = np.exp(score)
  sum_score = np.sum(score, axis = 1)
  loss = -1 * np.mean(np.log(score[range(num_train), y] / sum_score)) + reg *\
         np.sum(W * W)
  dS = score / sum_score.reshape(num_train, 1) 
  dS[range(num_train), y] -= 1
  dW = X.T.dot(dS) / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
