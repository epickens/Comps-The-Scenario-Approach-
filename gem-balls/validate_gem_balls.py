import numpy as np
import math
import scipy.stats as st


def validate_gem(kp, kn, Np, Nn, betap, betan):

    if kp < 0 or kp > Np + 1 or kn < 0 or kn > Nn + 1:
        raise ValueError('wrong input values')

    if kp == 0:
        errp = 0
    elif kp == Np + 1:
        errp = 1
    else:
        errp = st.beta.ppf(1 - betap / Np, kp, Np - kp + 1)


    if kn == 0:
        errn = 0
    elif kn == Nn + 1:
        errn = 1
    else:
        errn = st.beta.ppf(1 - betan / Nn, kn, Nn - kn + 1)

    return [errp, errn]

