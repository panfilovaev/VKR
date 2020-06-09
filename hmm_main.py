import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


class HMM:
    def __init__(self, M, K):
        self.M = M  # number of hidden states
        self.K = K  # number of Gaussians

    def fit(self, X, learning_rate=1e-2, max_iter=10, keep=False):
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        N = len(X)
        D = X[0].shape[1]  # assume each x is organized (T, D)

        if not keep:
            pi0 = np.ones(self.M)  # initial state distribution
            A0 = np.random.randn(self.M, self.M)  # state transition matrix
            R0 = np.ones((self.M, self.K))  # mixture proportions
            mu0 = np.zeros((self.M, self.K, D))
            for i in range(self.M):
                for k in range(self.K):
                    random_idx = np.random.choice(N)
                    x = X[random_idx]
                    random_time_idx = np.random.choice(len(x))
                    mu0[i, k] = x[random_time_idx]
            sigma0 = np.random.randn(self.M, self.K, D, D)
            thx, cost = self.set(pi0, A0, R0, mu0, sigma0)
        else:
            thx, cost = self.set(self.preSoftmaxPi.get_value(), self.preSoftmaxA.get_value(),
                                 self.preSoftmaxR.get_value(), self.mu.get_value(), self.sigmaFactor.get_value())

        pi_update = self.preSoftmaxPi - learning_rate * T.grad(cost, self.preSoftmaxPi)
        A_update = self.preSoftmaxA - learning_rate * T.grad(cost, self.preSoftmaxA)
        R_update = self.preSoftmaxR - learning_rate * T.grad(cost, self.preSoftmaxR)
        mu_update = self.mu - learning_rate * T.grad(cost, self.mu)
        sigma_update = self.sigmaFactor - learning_rate * T.grad(cost, self.sigmaFactor)

        updates = [
            (self.preSoftmaxPi, pi_update),
            (self.preSoftmaxA, A_update),
            (self.preSoftmaxR, R_update),
            (self.mu, mu_update),
            (self.sigmaFactor, sigma_update),
        ]

        train_op = theano.function(
            inputs=[thx],
            updates=updates,
        )

        costs = []
        for it in range(max_iter):
            print("it:", it)
            
            c = self.log_likelihood_multi(X).sum()
            print("c:", c)
            costs.append(c)
            
            for n in range(N):
                             
                train_op(X[n])

        return costs

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxR, mu, sigmaFactor):
        self.preSoftmaxPi = theano.shared(preSoftmaxPi)
        self.preSoftmaxA = theano.shared(preSoftmaxA)
        self.preSoftmaxR = theano.shared(preSoftmaxR)
        self.mu = theano.shared(mu)
        self.sigmaFactor = theano.shared(sigmaFactor)
        M, K = preSoftmaxR.shape
        self.M = M
        self.K = K

        pi = T.nnet.softmax(self.preSoftmaxPi).flatten()
        A = T.nnet.softmax(self.preSoftmaxA)
        R = T.nnet.softmax(self.preSoftmaxR)

        D = self.mu.shape[2]
        twopiD = (2 * np.pi) ** D

        # set up theano variables and functions
        thx = T.matrix('X')  # represents a TxD matrix of sequential observations

        def mvn_pdf(x, m, S):
            k = 1 / T.sqrt(twopiD * T.nlinalg.det(S))
            e = T.exp(-0.5 * (x - m).T.dot(T.nlinalg.matrix_inverse(S).dot(x - m)))
            return k * e

        def gmm_pdf(x):
            def state_pdfs(xt):
                def component_pdf(j, xt):
                    Bj_t = 0
                    # j = T.cast(j, 'int32')
                    for k in range(self.K):
                        # k = int(k)
                        # a = R[j,k]
                        # b = mu[j,k]
                        # c = sigma[j,k]
                        L = self.sigmaFactor[j, k]
                        S = L.dot(L.T)
                        Bj_t += R[j, k] * mvn_pdf(xt, self.mu[j, k], S)
                    return Bj_t

                Bt, _ = theano.scan(
                    fn=component_pdf,
                    sequences=T.arange(self.M),
                    n_steps=self.M,
                    outputs_info=None,
                    non_sequences=[xt],
                )
                return Bt

            B, _ = theano.scan(
                fn=state_pdfs,
                sequences=x,
                n_steps=x.shape[0],
                outputs_info=None,
            )
            return B.T

        B = gmm_pdf(thx)

        # scale = T.zeros((thx.shape[0], 1), dtype=theano.config.floatX)
        # scale[0] = (self.pi*B[:,0]).sum()

        def recurrence(t, old_a, B):
            a = old_a.dot(A) * B[:, t]
            s = a.sum()
            return (a / s), s

        [alpha, scale], _ = theano.scan(
            fn=recurrence,
            sequences=T.arange(1, thx.shape[0]),
            outputs_info=[pi * B[:, 0], None],
            n_steps=thx.shape[0] - 1,
            non_sequences=[B],
        )

        cost = -T.log(scale).sum()
        self.cost_op = theano.function(
            inputs=[thx],
            outputs=cost,
        )

        cost_detect = -T.log(scale)
        self.cost_detect = theano.function(
            inputs=[thx],
            outputs=cost_detect,
        )
        return thx, cost

    def log_likelihood_multi(self, X):
        return np.array([self.cost_op(x) for x in X])

    def log_likelihood_detect(self, X):
        return np.array(self.cost_detect(X))