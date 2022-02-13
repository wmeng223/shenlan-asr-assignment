# Author: Sining Sun , Zhanheng Yang

import numpy as np
from utils import *
import scipy.cluster.vq as vq

num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class GMM:
    def __init__(self, D, K=5):
        assert(D>0)
        self.dim = D
        self.K = K
        #Kmeans Initial
        self.mu , self.sigma , self.pi = self.kmeans_initial()

    def kmeans_initial(self):
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        # centroids: cluster center vector with 39 dim
        # labels: per frame labels
        #print(centroids, type(centroids))
        #print(labels, len(labels))
        clusters = [[] for i in range(self.K)] # K class (all training data)
        for (l,d) in zip(labels,data):
            clusters[l].append(d)

        for cluster in clusters: # K in K-means equal to K gmms
            mu.append(np.mean(cluster, axis=0)) # calculate per class mu 5*39
            sigma.append(np.cov(cluster, rowvar=0)) # 5*(39*39) row:observations
        pi = np.array([len(c)*1.0 / len(data) for c in clusters])
        return np.array(mu) , np.array(sigma) , np.array(pi)
    
    def gaussian(self , x , mu , sigma):
        """Calculate gaussion probability.
    
            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        D=x.shape[0]
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + 0.0001)
        mahalanobis = np.dot(np.transpose(x-mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x-mu))
        const = 1/((2*np.pi)**(D/2))
        return const * (det_sigma)**(-0.5) * np.exp(-0.5 * mahalanobis)
    
    def calc_log_likelihood(self , X):
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model 
        """

        log_llh = 0.0
        for i in range(X.shape[0]):
            tmp = 0.0
            for j in range(self.K):
                tmp += self.pi[j]*self.gaussian(X[i], self.mu[j], self.sigma[j])
            log_llh += np.log(tmp)
        return log_llh

    def em_estimator(self , X):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model 
        """

        log_llh = 0.0
        #gamma = [np.zeros(self.K) for i in range(X.shape[0])]
        N = X.shape[0]
        #print(N)
        gamma = np.zeros((N, self.K)) # N*K
        # step E, update gamma
        for i in range(N):
            prob = np.array([[self.pi[k]*self.gaussian(X[i], self.mu[k], self.sigma[k]) for k in range(self.K)]])
            gamma[i,:] = prob / np.sum(prob)
        #print(gamma, gamma.shape)
        # step M
        #mu = np.zeros((self.K, self.dim))
        #pi = np.zeros(self.K)
        sigma = np.zeros((self.K, self.dim, self.dim))
        Nk = np.sum(gamma, axis=0) # 1*K
        self.pi = Nk/N #update pi
        for k in range(self.K):
            self.mu[k,:] = np.dot(gamma[:,k].T, X) /Nk[k] # 39dim vector, update mu[k]
            # print(self.mu)
            for n in range(N):
                sigma[k,:]+=gamma[n,k]*np.outer(X[n]-self.mu[k], X[n]-self.mu[k])
            sigma[k,:] /= Nk[k] # update sigma[k]
        
        self.sigma = sigma
        #print(self.sigma)
        log_llh = self.calc_log_likelihood(X)

        return log_llh


def train(gmms, num_iterations = num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)   #
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
    return gmms

def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian) #Initial model
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()
    #data = read_all_data('train/feats.scp')
    #print(data, type(data), data.shape)
    #gmm1=GMM(39, K=num_gaussian)
    #print(gmm1.mu, gmm1.mu.shape)
    #print(gmm1.sigma, gmm1.sigma.shape)
    #print(gmm1.pi, type(gmm1.pi))
    #print(gmm1.calc_log_likelihood(data), gmm1.calc_log_likelihood(data).shape)
    #gmm1.em_estimator(data)