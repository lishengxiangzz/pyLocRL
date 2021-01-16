import numpy as np
import pydsp.thinkdsp as thindsp
import matplotlib.pyplot as plt
from transmitter import Transmitter
from receiver import Receiver


class Channel(object):
    def __init__(self, tran, recs):
        """
        Generate a free space loss channel from the transmitter  to the receivers
        :param tran: transmitter defined by class Transmitter
        :param recs: receivers defined by class Receiver
        """
        self.tran = tran
        self.recs = recs
        self.dists2trans = [np.sqrt(np.square(rec.x - tran.x)
                                    + np.square(rec.y - tran.y))
                            for rec in recs]
        self.attenuations = []
        self.delays = []

    def propagate(self, duration, Rs, symbol_wave_freq, carrier_freq, fs, amp, noise_amp, wave_template='sine',
                  data_seed=2021, noise_seed=2102):
        """
        propagate the signal transmitted by the transmitter to all the receivers
        :param duration: duration of the signal to receive
        :param Rs:
        :param symbol_wave_freq:
        :param fs:
        :param amp:
        :param wave_template:
        :param seed: seed to generate the symbol data
        :return:
        """
        c = 3e8
        d_min = np.min(self.dists2trans)
        d_max = np.max(self.dists2trans)
        offsets = (np.array(self.dists2trans) - d_min) / c
        offsets_samples = (offsets * fs).astype(int)
        duration_samples = int(duration * fs)
        duration_total = duration + (d_max - d_min) / c
        symbol_num = (np.ceil(duration_total * Rs)).astype(int)

        np.random.seed(data_seed)
        data = (2 * (np.random.randint(0, 2, symbol_num) - 0.5)).astype(int)
        self.tran.symbol_wave_gen(Rs=Rs, symbol_wave_freq=symbol_wave_freq, amp=amp, wave_template=wave_template)
        wave_mod = self.tran.modulate(data=data, fs=fs)

        # free space loss
        self.attenuations = [c / (4 * np.pi * dist * carrier_freq) for dist in self.dists2trans]
        # print("attenuations:", self.attenuations)

        for i in range(len(self.recs)):
            sig_i = self.tran.symbol_wave.make_wave(duration=duration, start=0, framerate=fs)
            sig_i.ys[:duration_samples] = wave_mod.ys[offsets_samples[i]:offsets_samples[i] + duration_samples]
            sig_i.ys *= self.attenuations[i]

            # plt.plot(sig_i.ys)
            # plt.show()

            # add noises
            np.random.seed(noise_seed+21*i)
            noises = np.random.normal(0, scale=noise_amp, size=len(sig_i.ys))
            sig_i.ys += noises
            # plt.plot(sig_i.ys)
            # plt.show()

            self.recs[i].signal = sig_i
            self.recs[i].SNR = 0.5*np.square(self.attenuations[i])/np.square(noise_amp)

        return self.recs


if __name__ == "__main__":
    transmitter = Transmitter([0, 0])
    receiver1 = Receiver([-3000, 0], is_center=True)
    receiver2 = Receiver([-3300, 0], is_center=False)
    receiver3 = Receiver([-3600, 0], is_center=False)
    channel = Channel(transmitter, [receiver1, receiver2, receiver3])

    recs = channel.propagate(duration=1e-5, Rs=1e6, symbol_wave_freq=1e6, fs=1e8, wave_template='sine')

    for i in range(len(recs)):
        recs[i].signal.plot()
        plt.show()

