import time
from src.utils import *

from numpy.linalg import norm
from numpy.linalg import pinv

from scipy.sparse.linalg import cg

def pos(x):
    return (x > 0) * x

def CG_solver(A, B):
    Q = None
    for i in B:
        x = cg(A, i)

        if isinstance(Q, type(None)) == True:
            Q = x[0]
        else:

            Q = np.c_[Q, x[0]]

    return Q

def nls_subproblem(X, W, H_init, tol, max_iter):
    H = H_init
    WtX = np.dot(W.T, X)
    WtW = np.dot(W.T, W)

    alpha = 1
    beta = 0.1

    for i in range(max_iter):
        grad = np.dot(WtW, H) - WtX
        proj_gradient = norm(grad[np.logical_or(grad < 0, H > 0)])
        if proj_gradient < tol:
            break

        for j in range(20):
            Hn = H - alpha * grad
            Hn = pos(Hn)
            d = Hn - H
            gradd = np.sum(grad * d)
            dQd = np.sum(np.dot(WtW, d) * d)
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0
            if j == 0:
                decr_alpha = not suff_decr
                Hp = H

            if decr_alpha:
                if suff_decr:
                    H = Hn
                    break
                else:
                    alpha *= beta
            elif not suff_decr or (Hp == Hn).all():
                H = Hp
                break
            else:
                alpha /= beta
                Hp = Hn

    return H, grad, i

class OTDA(object):

    def __init__(
            self, numterms, n_topic,
            alpha=1.0, beta=0.1, max_iter=100, max_err=1e-3,
            fix_seed=False):

        if fix_seed:
            np.random.seed(0)

        self.numterms = numterms
        self.n_topic = n_topic
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.B = np.ones([self.n_topic, 1])
        self.max_err = max_err

        self.loss = 0

        self.nmf_init_rand()

        self.U1 = np.zeros((self.numterms, self.n_topic))
        self.V1 = np.zeros((self.n_topic, self.n_topic))

        self.U2 = np.zeros((self.numterms, self.n_topic))
        self.V2 = np.zeros((self.n_topic, self.n_topic))


    def nmf_init_rand(self):
        self.W1 = np.random.random((self.numterms, self.n_topic))
        self.W2 = np.random.random((self.numterms, self.n_topic))


    def nmf_iter(self, A, S, H_init, alpha, stream_iter=None):
        loss_old = 1e20
        self.loss = 0
        if stream_iter != None:
            self.max_iter = stream_iter

        start_time = time.time()
        if self.max_iter == 0:
            loss = self.nmf_loss(A, S, H_init)

        for i in range(self.max_iter):
            H_init = self.nmf_solver(A, S, H_init, alpha)
            loss = self.nmf_loss(A, S, H_init)
            if loss_old - loss < self.max_err:
                print('loss converged')
                break
            loss_old = loss
            end_time = time.time()
            print('Step={}, Loss={}, Time={}s'.format(i,loss, end_time - start_time))

        self.loss = loss

        self.U1 = self.U1 * alpha
        self.V1 = self.V1 * alpha
        self.U2 = self.U2 * alpha
        self.V2 = self.V2 * alpha
        self.W1 = self.W1 * alpha
        self.W2 = self.W2 * alpha

        return H_init

    def nmf_solver(self, A, S, H_init, alpha):

        HT, _, _ = nls_subproblem(A, self.W1, H_init.T, 1e-3, 1000)
        H = HT.T

        self.U2 = self.U2 + np.dot(S.T, self.W1)
        self.V2 = self.V2 + np.dot(self.W1.T, self.W1)
        gradW2 = self.U2 - np.dot(self.W2, self.V2)

        V_inv = pinv(self.V2)
        self.W2 += np.dot(gradW2, V_inv)
        # Q = CG_solver(self.V1.T, gradW2)
        # self.W1 += Q.T
        self.W2 = pos(self.W2)

        AS = np.r_[A.T, np.sqrt(self.alpha) * S.T]
        HW2 = np.r_[H, np.sqrt(self.alpha) * self.W2]
        self.U1 = self.U1 + np.dot(AS.T, HW2)
        self.V1 = self.V1 + np.dot(HW2.T, HW2)
        gradW1 = self.U1 - np.dot(self.W1, self.V1)

        V_inv = pinv(self.V1)
        self.W1 += np.dot(gradW1, V_inv)
        # Q = CG_solver(self.V1.T, gradW1)
        # self.W1 += Q.T
        self.W1 = pos(self.W1)

        return H

    def nmf_loss(self, A, S, H):
        loss = norm(A - np.dot(self.W1, np.transpose(H)), 'fro') ** 2 / 2.0
        if self.alpha > 0:
            temp = loss
            loss += self.alpha * norm(np.dot(self.W1, np.transpose(self.W2)) - S, 'fro') ** 2 / 2.0
        if self.beta > 0:
            loss += self.beta * norm(H, 1) ** 2 / 2.0

        return loss

    def get_lowrank_matrix(self):
        return self.W1, self.W2


    def save_model(self, outputfolder):
        obj_list = [self.W1, self.W2, self.H]
        save_objects(obj_list, outputfolder + 'obj/otda.pkl')

    def load_model(self, outputfolder):
        self.W1, self.W2, self.H = load_objects(outputfolder + 'obj/otda.pkl')# -*- coding: utf-8 -*-

