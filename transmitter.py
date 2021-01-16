import pydsp.thinkdsp as thinkdsp
import numpy as np


class Transmitter(object):
    def __init__(self, pos):
        """
        Radio transmitter.

        :param x0: x-coordinate of the transmitter

        :param y0: y-coordinate of the transmitter

        :param Rs: Symbol rate (Hz)

        :param amp: Amplitude of the signal

        :param carrier_fs: carrier frequency

        """
        self.x = pos[0]
        self.y = pos[1]
        self.pos = pos
        self.Rs = []
        self.symbol_wave_freq = []
        self.amp = []
        self.data = []
        self.symbol_wave = []

    def symbol_wave_gen(self, Rs, symbol_wave_freq, amp=1, wave_template='sine'):
        """
        generate the symbol wave, which is in the form of Sine, the carrier frequency is defined by carrier_fs
        :param: wave_template: the template of symbol wave, 'sine', 'square', 'triangular'
        :return:
        """
        self.Rs = Rs
        self.symbol_wave_freq = symbol_wave_freq
        self.amp = amp
        if wave_template == 'square':
            self.symbol_wave = thinkdsp.SquareSignal(freq=self.symbol_wave_freq, amp=self.amp, offset=0)
        elif wave_template == 'triangular':
            self.symbol_wave = thinkdsp.TriangleSignal(freq=self.symbol_wave_freq, amp=self.amp, offset=0)
        else:
            self.symbol_wave = thinkdsp.SinSignal(freq=self.symbol_wave_freq, amp=self.amp, offset=0)
        return self.symbol_wave

    def modulate(self, data, fs):
        """
        Modulate the given data with the symbol wave
        :param data: data to be modulated, {-1,1}
        :param fs: sample rate of the receiver
        :return:
        """
        self.data = data
        sampled_sym_wave = self.symbol_wave.make_wave(duration=1/self.Rs, start=0, framerate=fs)
        temp_wave = np.kron(data, sampled_sym_wave.ys)
        sampled_mod_wave = self.symbol_wave.make_wave(duration=len(data)*1/self.Rs, start=0, framerate=fs)
        sampled_mod_wave.ys = temp_wave

        return sampled_mod_wave



#   test
import matplotlib.pyplot as plt

if __name__ == '__main__':
    transmitter = Transmitter([0, 0])
    transmitter.symbol_wave_gen(wave_template='sine', Rs=1e6, symbol_wave_freq=1e6, amp=1)

    fs = 1e8
    temp_wave = transmitter.symbol_wave.make_wave(duration=1/transmitter.Rs, start=0, framerate=fs)
    temp_wave.plot()
    plt.show()

    np.random.seed(2021)
    symbol_num = 10
    data = (2 * (np.random.randint(0, 2, symbol_num) - 0.5)).astype(int)

    mod_wave = transmitter.modulate(data=data, fs=fs)

    mod_wave.plot()

    plt.show()
