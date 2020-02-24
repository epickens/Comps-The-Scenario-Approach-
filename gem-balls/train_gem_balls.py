import numpy as np
import math


def train_gem(X, Y, cp=1, cn=1, supp_cap=1):
    lab1 = 1
    lab2 = 0
    labs = [1,0]

    lab1_ind = np.argwhere(Y==lab1)
    lab2_ind = np.argwhere(Y==lab2)

    Np = lab1_ind.shape[0]
    Nn = lab2_ind.shape[0]

    if Np + Nn != Y.shape[0]:
        raise Exception('Y should contain only the labels 0 and 1')

    #reset values to allow for the use of the mod function
    Y[lab1_ind] = 1
    Y[lab2_ind] = 2
    c = [cp, cn]
    s1 = []
    s2 = []
    S = {}
    S['1'] = s1
    S['2'] = s2

    xc = X[1,:]
    yc = Y[1]

    R = X[2:,:]
    L = Y[2:]

    classifier = []
    complete = 0

    while complete == 0:
        opp_label = (yc % 2) + 1

        dist = np.sqrt(np.sum(np.subtract(R, xc) ** 2, axis=1))
        sorted_dist_ind = np.argsort(dist)
        dist = np.sort(dist)

        R = R[sorted_dist_ind,:]
        L = L[sorted_dist_ind]

        opp_pos = np.argwhere(L == opp_label)
        supp_pos = opp_pos[0:min(c[opp_label-1], opp_pos.shape[0])]
        supps = S[str(opp_label)]

        if c[opp_label-1] > opp_pos.shape[0]:
            complete = 1

            classifier.append([xc, math.inf, labs[yc-1]])
            supps.append(np.full((1, xc.shape[0]), math.inf))

        else:

            classifier.append([xc, dist[supp_pos[-1]], labs[yc-1]])
            supps.append(R[supp_pos,:])

            temp_start = supp_pos[-1][0]
            len_rl = R.shape[0]

            xc = R[temp_start,:]
            yc = L[temp_start]

            R = R[temp_start+1:len_rl,:]
            L = L[temp_start+1:len_rl]

        S[str(opp_label)] = supps

    kp = len(S['1'])
    kn = len(S['2'])

    return [classifier, kp, kn, Np, Nn]

