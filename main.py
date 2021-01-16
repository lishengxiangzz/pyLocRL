# This is a environment for passive location.

import misc
import matplotlib.pyplot as plt
import numpy as np
from transmitter import Transmitter
from channel import Channel
from receiver import Receiver
from LocEnvs import LocationEnv
import pickle

Rs = 1e6    # symbol rate
symbol_wave_freq = 1e6  # the symbol wave is in the form of "sine", the frequency of which is symbol_wave_freq
carrier_freq = 2400e6   # only used in calculate the free space loss, together with the distance
fs = 100e6  # sample rate adopted by the receivers, 100MHz
amp_trans = 1
noise_amp = 20e-6    # noise added to the signal, the same with all the map
duration = 100e-6
data_seed = 2021
noise_seed = 178

# misc.show_geometry(transmitter, receivers)


if __name__ == '__main__':

    dis_rmses = []
    snrs = []
    for loc in np.linspace(5000,1000, 100):
        transmitter = Transmitter([0, 0])
        receiver1 = Receiver([-loc*0.7, loc*0.7], is_center=True)
        receiver2 = Receiver([loc*0.7, loc*0.7], is_center=False)
        receiver3 = Receiver([0, -loc], is_center=False)
        receivers = [receiver1, receiver2, receiver3]
        # misc.show_geometry(transmitter, receivers)
        locationEnv = LocationEnv(tran=transmitter, recs=receivers, duration=duration, Rs=Rs,
                                  symbol_wave_freq=symbol_wave_freq, carrier_freq=carrier_freq,
                                  fs=fs, amp=amp_trans, noise_amp=noise_amp, data_seed=data_seed,
                                  noise_seed=noise_seed)
        dis_rmse, snr = locationEnv.step()
        print(loc,'km: ',dis_rmse)
        dis_rmses.append(dis_rmse)
        snrs.append(snr)
    variables = {"dis_rmse":dis_rmses,"snrs":snrs}
    with open('dumps/varibles.plk', 'wb') as file:
        pickle.dump(variables, file)

    plt.plot(dis_rmses)
    plt.show()

