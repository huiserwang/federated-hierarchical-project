import numpy as np
from numpy import random
import math

def Unm(d,p,n,m):

    B = 1
    D = 20
    M = 5
    N = 6
    U = 5
    Y = 1
    alpha = 0.2
    beta = 1
    sigma = 1
    ceta = 2
    epsilon = 0.5
    fr = 5
    sumbd = 0
    Pt = 125

    C_cmpv = np.zeros((N,M))
    C_cmpr = np.zeros(M)
    epslv = np.zeros((N,M))
    epslr = np.zeros(M)
    C_gFL = np.zeros((N,M))
    C_tFL = np.zeros(M)
    h = np.ones((N,M))
    Rv = np.zeros((N,M))
    C_totv = np.zeros(N)
    C_totr = np.zeros(M)
    Rr = np.zeros(M)

    C_cmpv[n][m] = ceta * U * d[n][m] * fr * fr
    C_cmpr[m] = ceta * U * D * fr * fr
    L = - math.log(1 - epsilon,10)
    epslv[n][m] = alpha * math.log(1 + d[n][m],10)
    C_gFL[n][m] = - math.log(1 - epslv[n][m],10) * C_cmpv[n][m]
    for i in range(1,N):
        sumbd = sumbd + beta * d[i][m]
    epslr[m] = alpha * math.log(1 + D + sumbd,10)
    C_tFL[m] = - math.log(1 - epslr[m],10) * C_cmpr[m]
    for j in range(1,M):
        Rv[n][j] = B * math.log(1 + Pt * h[n][j] / sigma / sigma,10)
        C_totv[n] = C_totv[n] + C_gFL[n][j] + Pt * d[n][j] / Rv[n][j]
    for i in range(1,N):
        Rr[m] = Rr[m] + B * math.log(1 + Pt * h[i][m] / sigma / sigma,10)
    C_totr[m] = Pt * Y / Rr[m] + C_tFL[m]
    Un = C_totv[n]
    Um = C_totr[m]
    for j in range(1,M):
        Un = Un - p[j][n] * d[n][j] / np.sum(d)
        Um = Um - p[j][n] * d[n][j] / np.sum(d)

    return Un,Um

###main函数
def get_d_matrix(N=6, M=5, rho=1, scale=5):
    d = random.randint(1,scale,size = (N,M))
    p = random.randint(1,scale,size = (M,N))
    z = d
    mu = np.zeros((N,M))
    n = 1
    m = 1
    Um = 0
    Umpast = 0
    h = 0
    iter = 5
    delta = 1

    for j in range(1,M):
        Untemp,Umtemp = Unm(d,p,n,j)
        Um = Um + Umtemp

    while(Umpast - Um < delta and h < iter):

        h = h + 1

        for j0 in range (1,M):

            for i in range(1,N):
                for k in range(1,5):

                    for i1 in range (1,scale):
                        for i2 in range (1,scale):
                            for i3 in range (1,scale):
                                for i4 in range (1,scale):
                                    for i5 in range (1,scale):
                                        Unpresent,Umpresent = Unm(d,p,i,j0)
                                        dnpresenttemp = 0
                                        for j in range (1,M):
                                            dnpresenttemp = dnpresenttemp + (d[i][j] - z[i][j] + mu[i][j]) * (d[i][j] - z[i][j] + mu[i][j])
                                        dnpresent = Unpresent + rho / 2 * dnpresenttemp

                                        dnext = d
                                        dnext[i] = [i1,i2,i3,i4,i5]
                                        Unnext,Umnext = Unm(dnext,p,i,j0)
                                        dnnexttemp = 0
                                        for j in range (1,M):
                                            dnnexttemp = dnnexttemp + (dnext[i][j] - z[i][j] + mu[i][j]) * (dnext[i][j] - z[i][j] + mu[i][j])
                                        dnnext = Unnext + rho / 2 * dnnexttemp

                                        if dnpresent > dnnext:
                                            d = dnext

                    for i1 in range (1,scale):
                        for i2 in range (1,scale):
                            for i3 in range (1,scale):
                                for i4 in range (1,scale):
                                    for i5 in range (1,scale):
                                        znpresent = 0
                                        for j in range (1,M):
                                            znpresent = znpresent + (d[i][j] - z[i][j] + mu[i][j]) * (d[i][j] - z[i][j] + mu[i][j])

                                        znext = z
                                        znext[i] = [i1,i2,i3,i4,i5]
                                        znnext = 0
                                        for j in range (1,M):
                                            znnext = znnext + (d[i][j] - znext[i][j] + mu[i][j]) * (d[i][j] - znext[i][j] + mu[i][j])

                                        if znpresent > znnext:
                                            z = znext
                    
                    mu[i] = d[i] - z[i] + mu[i]
            
            Umpast = Um
            pnext = random.randint(1,scale,size = (M,N))
            if Unm(d,p,n,j0) > Unm(d,pnext,n,j0):
                p = pnext
            for j in range(1,M):
                Untemp,Umtemp = Unm(d,p,n,j)
                Um = Um + Umtemp
    return d