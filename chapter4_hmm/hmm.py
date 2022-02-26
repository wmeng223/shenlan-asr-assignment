# Author: Kaituo Xu, Fan Yu
import numpy as np

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O) # observations
    N = len(pi) # states
    prob = 0.0
    
    # Begin Assignment
    alphas = np.zeros((N, T))
    # Put Your Code Here
    for t in range(T):
        for i in range(N):
            if t == 0:
                alphas[i][t]=pi[i]*B[i][O[t]]
            else:
                #alphas[i][t] = np.dot([alpha[t-1] for alpha in alphas], [a[i] for a in A]) * B[i][O[t]]
                alphas[i][t]=np.dot(alphas[:,t-1], np.array(A)[:,i]) * np.array(B)[i][O[t]]
    # End Assignment
    return np.sum(alphas[:,N-1])


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment
    beta = np.zeros((N,T))
    # Put Your Code Here
    for t in range(T-1,-1,-1):
        for i in range(N):
            if t == T-1:
                beta[i][t]=1
            else:
                tmp = 0
                for j in range(N):
                    tmp += A[i][j]*B[j][O[t+1]]*beta[j][t+1]
                beta[i][t]=tmp
    for i in range(N):
        prob += pi[i]*B[i][O[0]]*beta[i][0]
    # End Assignment
    return prob


def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, [0] * T
    # Begin Assignment
    delta = np.zeros((N, T))
    Psi = np.zeros((N, T))
    # Put Your Code Here
    for t in range(T):
        for i in range(N):
            if t == 0:
                delta[:,0] = np.array(pi).T * np.array(B)[:,O[0]]
            else:
                tmp_list = []
                for j in range(N):
                    tmp_list = np.append(tmp_list, delta[j][t-1]*A[j][i])
                delta[i][t] = max(tmp_list)*B[i][O[t]]
                Psi[i][t] = np.argmax(tmp_list)
                print('t=%s ' % t, tmp_list)
    # Stop
    print('delta:\n', delta)
    print('Psi:\n', Psi)
    best_prob = max(delta[:,T-1])
    best_cur_state = np.argmax(delta[:,T-1])
    print(best_cur_state)
    best_path[T-1]=best_cur_state
    
    for t in range(T-2,-1,-1):
        #print(best_cur_state)
        best_path[t] = int(Psi[best_path[t+1]][t+1])
        print(best_path[t])
        #best_path.append(best_cur_state)
    # End Assignment
    best_path = [val+1 for val in best_path]
    return best_prob, best_path


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0) # red=0 white=1
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward) # 0.130218
    
    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward) # 0.130218
    
    best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    print(best_prob, best_path)
