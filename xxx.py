import pickle
import matplotlib.pyplot as plt
import numpy as np


variables =  []
with open('dumps/varibles.plk','rb') as file:
    variables = pickle.load(file)

dis_rmses, snrs = variables['dis_rmse'], variables['snrs']
print(dis_rmses)
plt.plot(np.linspace(5000,1000,len(dis_rmses)), 10*np.log10(dis_rmses))
plt.legend(["RMSE(dB)"])
# plt.plot(np.linspace(5000,1000,len(dis_rmses)), dis_rmses)
# plt.legend(["RMSE"])
# plt.ylim([0,20])
plt.grid(True)
plt.show()

print(snrs)
snrs = np.vstack(snrs)
plt.plot(np.linspace(5000,1000,len(snrs)), 10*np.log10(snrs))
plt.legend(["SNR1","SNR2","SNR3"])
plt.grid(True)
plt.show()