import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from transmitter import Transmitter
from receiver import Receiver


def lag_estimate(data1, data2, fs):
    '''
    Estimate the lagged time of data1 to data2, if lag<0, data1 appears first
    :param data1:
    :param data2:
    :param fs: the sample rate (Hz)
    :return: the estimation of lag time
    '''

    data1 = data1[::-1]

    conv = np.abs(np.convolve(data1, data2))
    # plt.plot(conv)
    # plt.show()
    conv_peak_index = conv.argmax()
    lag_samples = conv_peak_index + 1 - len(data1)
    lag_time = lag_samples / fs
    return lag_time, lag_samples


def show_geometry(transmitter, stations):
    plt.figure(figsize=(10, 10))
    plt.scatter(transmitter.x, transmitter.y, marker='s', s=[600], c=['r'])

    x = [station.x for station in stations]
    y = [station.y for station in stations]
    plt.scatter(x, y, marker='o', s=[400] * len(stations), c=['b', 'g', 'k'])
    plt.legend(['transmitter', 'receivers'], markerscale=0.3)
    plt.grid(True)
    plt.xlim([-5000,5000])
    plt.ylim([-5000,5000])
    plt.show()


def eucli_dist(a, b):
    return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))


def residual(p, pairs, delta_t):
    c = 3e8
    res = [eucli_dist(p, pi[0]) - eucli_dist(p, pi[1]) - delta_t_i * c for (pi, delta_t_i) in zip(pairs, delta_t)]
    return res


def position_solver(pairs, delta_t):
    """
    solve the position for given receiver pairs and corresponding delays
    :param pairs: [[p_i,p_j]],
    :param delta_t: [delta1,]
    :return:
    """
    r = optimize.leastsq(residual, [500, 500], args=(pairs, delta_t), maxfev=100000)
    return r

def pos_dist(p1, p2):
    return np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))

def CRLB(receivers, tran):
    """
    Calculate the CRLB with received signal together with the geometric configuration
    :param receivers:
    :return:
    """
    # CRLB = (G_0^T*Q^{-1}*G_0 )^{-1}
    # construct G_0
    r1 = pos_dist(receivers[0].pos, tran.pos)
    r2 = pos_dist(receivers[1].pos, tran.pos)
    r3 = pos_dist(receivers[2].pos, tran.pos)


    sigma = np.power(r1*r2*r3,2/3)
    Q = np.array([[2 * sigma, sigma],
                    [sigma, 2 * sigma]])
    # Q = np.array([[5000, 2500],
    #               [2500, 5000]])

    G_0 = np.array([[(receivers[0].x-tran.x)/r1-(receivers[1].x-tran.x)/r2, (receivers[0].y-tran.y)/r1-(receivers[1].y-tran.y)/r2],
                      [(receivers[0].x-tran.x)/r1-(receivers[2].x-tran.x)/r3, (receivers[0].y-tran.y)/r1-(receivers[2].y-tran.y)/r3]])


    Q_pinv = np.linalg.pinv(Q)

    crlb = np.linalg.pinv(G_0.transpose() @ Q_pinv @ G_0)

    return crlb


if __name__ == '__main__':
    transmitter = Transmitter([0, 0])

    crlbs =[]

    for loc in range(5000, 1000, -100):
        receiver1 = Receiver([-loc, loc], is_center=True)
        receiver2 = Receiver([loc, loc], is_center=False)
        receiver3 = Receiver([loc, -loc], is_center=False)
        receivers = [receiver1, receiver2, receiver3]
        crlb = CRLB(receivers, transmitter)

        crlbs.append(np.sqrt(np.trace(crlb)/crlb.shape[0]))

    print(crlbs)
    plt.plot(crlbs)
    plt.show()





