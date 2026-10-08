import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/spectra/RR-Ab.txt')
print(data.shape)
data_mean = np.mean(data[:, 1:], axis=1)*0
for i in range(1, data.shape[1]):
    plt.plot(data[:, 0], data[:, i] - data_mean)
plt.title('RR-UV-Vis')
plt.show()

data = np.loadtxt('data/spectra/RR-CD.txt')
print(data.shape)
data_mean = np.mean(data[:, 1:], axis=1)*0
for i in range(1, data.shape[1]):
    plt.plot(data[:, 0], data[:, i] - data_mean)
plt.title('RR-CD')
plt.show()

# ----------------------------------------------------------

data = np.loadtxt('data/spectra/RS-Ab.txt')
print(data.shape)
data_mean = np.mean(data[:, 1:], axis=1)*0
for i in range(1, data.shape[1]):
    plt.plot(data[:, 0], data[:, i] - data_mean)
plt.title('RS-UV-Vis')
plt.show()

data = np.loadtxt('data/spectra/RS-CD.txt')
print(data.shape)
data_mean = np.mean(data[:, 1:], axis=1)*0
for i in range(1, data.shape[1]):
    plt.plot(data[:, 0], data[:, i] - data_mean)
plt.title('RS-CD')
plt.show()

# ----------------------------------------------------------

data_RR_Left = np.loadtxt('data/spectra/RR-fitted-left-256.txt')
data_RR_Right = np.loadtxt('data/spectra/RR-fitted-right-256.txt')
data_SR_Left = np.loadtxt('data/spectra/RS-fitted-left-256.txt')
data_SR_Right = np.loadtxt('data/spectra/RS-fitted-right-256.txt')

print(data_RR_Left.shape, data_RR_Right.shape, data_SR_Left.shape, data_SR_Right.shape)

data_mean = (np.mean(data_RR_Left[:, 1:], axis=1) +
             np.mean(data_RR_Right[:, 1:], axis=1) +
             np.mean(data_SR_Left[:, 1:], axis=1) +
             np.mean(data_SR_Right[:, 1:], axis=1))/4*0

for i in range(1, data.shape[1]):
    plt.plot(data_RR_Left[:, 0], data_RR_Left[:, i] - data_mean)
plt.title('RR-left')
plt.show()

for i in range(1, data.shape[1]):
    plt.plot(data_RR_Right[:, 0], data_RR_Right[:, i] - data_mean)
plt.title('RR-right')
plt.show()

for i in range(1, data.shape[1]):
    plt.plot(data_SR_Left[:, 0], data_SR_Left[:, i] - data_mean)
plt.title('RS-left')
plt.show()

for i in range(1, data.shape[1]):
    plt.plot(data_SR_Right[:, 0], data_SR_Right[:, i] - data_mean)
plt.title('RS-right')
plt.show()
