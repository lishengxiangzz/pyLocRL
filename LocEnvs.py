import misc
import matplotlib.pyplot as plt
import numpy as np
from transmitter import Transmitter
from channel import Channel
from receiver import Receiver


class LocationEnv(object):

    def __init__(self, tran, recs, duration, Rs, symbol_wave_freq, carrier_freq, fs, amp,
                 noise_amp, data_seed, noise_seed):
        self.tran = tran
        self.recs = recs
        self.channel = []
        self.duration = duration
        self.Rs = Rs
        self.symbol_wave_freq = symbol_wave_freq
        self.fs = fs
        self.data_seed = data_seed
        self.amp = amp
        self.noise_seed = noise_seed
        self.noise_amp = noise_amp
        self.carrier_freq = carrier_freq

    def pos_mse(self, receivers):
        lag_1to2t, lag_1to2s = misc.lag_estimate(receivers[0].signal.ys, receivers[1].signal.ys, fs=self.fs)
        lag_1to3t, lag_1to3s = misc.lag_estimate(receivers[0].signal.ys, receivers[2].signal.ys, fs=self.fs)
        lag_2to3t, lag_2to3s = misc.lag_estimate(receivers[1].signal.ys, receivers[2].signal.ys, fs=self.fs)
        # lag_2to3t, lag_2to3s = lag_1to2t - lag_1to3t, lag_1to2s - lag_1to3s

        # print('estimations: ', [lag_1to2s, lag_1to3s, lag_2to3s])
        # d1 = np.sqrt(np.square(self.tran.x - receivers[0].x) + np.square(self.tran.y - receivers[0].y))
        # d2 = np.sqrt(np.square(self.tran.x - receivers[1].x) + np.square(self.tran.y - receivers[1].y))
        # d3 = np.sqrt(np.square(self.tran.x - receivers[2].x) + np.square(self.tran.y - receivers[2].y))
        # print('check: ', np.array([d1 - d2, d1 - d3, d2 - d3]) / 3e8 * self.fs)

        pairs = [[receivers[0].pos, receivers[1].pos],
                 [receivers[0].pos, receivers[2].pos],
                 [receivers[1].pos, receivers[2].pos]]
        delta_t = [lag_1to2t, lag_1to3t, lag_2to3t]
        # print('estimated lags (s): ', delta_t)

        pos_solved = misc.position_solver(pairs, delta_t)

        # print('random seed: ', self.data_seed, self.noise_seed)
        # print("receivers' positions", [receivers[0].pos, receivers[1].pos, receivers[2].pos])
        # print('background transmitter pos: ', self.tran.pos)
        # print('estimated transmitter pos: ', pos_solved[0])

        error = misc.pos_dist(self.tran.pos, pos_solved[0])

        return error

    def step(self):
        self.channel = Channel(self.tran, self.recs)

        # for rec in receivers:
        #     rec.signal.plot()
        #     plt.show()

        receivers =[]
        dis = []
        est_num = 200
        for i in range(est_num):
            # Receive signals from the channel
            receivers = self.channel.propagate(duration=self.duration, Rs=self.Rs,
                                               symbol_wave_freq=self.symbol_wave_freq,
                                               carrier_freq=self.carrier_freq, fs=self.fs, amp=self.amp,
                                               noise_amp=self.noise_amp, wave_template='sine',
                                               data_seed=self.data_seed + i, noise_seed=self.noise_seed + i)

            dis_error = self.pos_mse(receivers)
            # print(dis_error)

            dis.append(dis_error)

        # mean of minimum 10 error
        dis = np.sort(dis)
        # print("soted:", dis)

        dis_error50 = dis[:int(est_num*0.68)] # (mu-sigma, mu+sigm) 2sigma: 0.9545
        # print(dis_error50[-1])

        return np.sqrt(np.mean(np.square(dis_error50))), [rec.SNR for rec in receivers ]
