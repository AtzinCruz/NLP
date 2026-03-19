#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
#
#  Modified to use CUDA cores via PyTorch for GPU-accelerated computation.

import numpy as np
import pylab
import torch

# Use CUDA if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"t-SNE running on: {device}")


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    # Compute pairwise distance matrix on GPU
    X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
    sum_X_gpu = torch.sum(X_gpu ** 2, dim=1)
    D_gpu = -2. * torch.mm(X_gpu, X_gpu.t()) + sum_X_gpu.unsqueeze(1) + sum_X_gpu.unsqueeze(0)
    D = D_gpu.cpu().numpy()

    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
        GPU-accelerated via PyTorch CUDA cores.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01

    # Compute P-values on CPU (binary search is sequential)
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.                              # early exaggeration
    P = np.maximum(P, 1e-12)

    # Move all tensors to GPU for the optimization loop
    P_gpu = torch.tensor(P, dtype=torch.float32, device=device)
    Y = torch.randn(n, no_dims, dtype=torch.float32, device=device)
    iY = torch.zeros(n, no_dims, dtype=torch.float32, device=device)
    gains = torch.ones(n, no_dims, dtype=torch.float32, device=device)

    # Run iterations entirely on GPU
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y ** 2, dim=1)
        num = -2. * torch.mm(Y, Y.t()) + sum_Y.unsqueeze(1) + sum_Y.unsqueeze(0)
        num = 1. / (1. + num)
        num.fill_diagonal_(0.)
        Q = num / torch.sum(num)
        Q = torch.clamp(Q, min=1e-12)

        # Compute gradient (vectorized — no Python loop over n)
        PQ = P_gpu - Q
        W = PQ * num  # W[j, i] = (P[j,i] - Q[j,i]) * num[j,i]
        col_sums = W.sum(dim=0)  # sum_j W[j,i] for each i
        dY = Y * col_sums.unsqueeze(1) - torch.mm(W.t(), Y)

        # Perform the update
        momentum = initial_momentum if iter < 20 else final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains = torch.clamp(gains, min=min_gain)
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - Y.mean(dim=0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P_gpu * torch.log(P_gpu / Q))
            print("Iteration %d: error is %f" % (iter + 1, C.item()))

        # Stop lying about P-values
        if iter == 100:
            P_gpu = P_gpu / 4.

    # Return solution as NumPy array
    return Y.cpu().numpy()


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()
