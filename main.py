import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import math
from enum import Enum

from matplotlib import pyplot


class Modulation(Enum):
    ASK = 'ASK'
    PSK = 'PSK'
    FSK = 'FSK'

# Params
sequence = '0110101000101011'
modulation = Modulation.ASK
M = 4
FREQ_PORTEUSE = 10
T_PORTEUSE = 1/FREQ_PORTEUSE
amp_porteuse = 1 # amp max de la porteuse
bande_passante = 6 # en hertz pour la fsk

POINTS_PER_PERIOD = 50



def modulate_ask(m, data):
    # Creer lookup table pour les bits
    amp_lookup_table = np.linspace(0, amp_porteuse, num=m, endpoint=True)
    amp_values = amp_lookup_table[data]
    sinValues = np.zeros(len(data) * POINTS_PER_PERIOD)
    timeValues = np.linspace(0, len(data) * T_PORTEUSE, num=POINTS_PER_PERIOD * len(data), endpoint=True)
    for i in range(len(data)):
        index1 = i * POINTS_PER_PERIOD
        index2 = index1 + POINTS_PER_PERIOD - 1
        sinValues[index1:index2] = amp_values[i] * np.sin(2 * math.pi * FREQ_PORTEUSE * timeValues[index1:index2])
    return timeValues ,sinValues

def modulate_bit(values, n): # Return a value of 0 -> m depending on the amount of bits per signal
    return [int(values[i:i + n], 2) for i in range(0, len(values), n)]


def modulate_psk():

    pass

def samples_per_period(frequency): # used mostly so we can have a good visualisation of the signal.
    samples = 20
    return samples * frequency

def modulate_fsk(m):

    pass

def is_m_valid(m):
    return (m > 0) and ((m & (m-1)) == 0)

def get_m_bit_amount(m):
    return math.log2(m)

def verify_inputs(sequence, modulation, M, SNR, bande_passante_HZ):
    seq_length = len(sequence)
    bit_amount = get_m_bit_amount(M)

    if seq_length % bit_amount != 0:
        print("Longeur de séquence invalide. La longeur de la séquence devrait être un multiple de n.")
        return False

    shannon = math.sqrt(1 + SNR)
    if M > shannon:
        print("Les paramêtres ne respectent pas le théorème de Shannon-Hartley.")
        return False

    return True

def generate_gaussian_noise(samples_count, std_dev):
    mean = 0.0  # Moyenne de la distribution centre sur 0
    # Generer le bruit gaussian
    return np.random.normal(loc=mean, scale=std_dev, size=samples_count)

def generate_ask_with_noise(num_bits=1000, amplitude=1.0, snr_db=10):
    # Step 1: Generate random bits (0 or 1)
    bits = np.random.randint(0, 2, num_bits)

    # Step 2: Generate ASK modulated signal
    signal = bits * amplitude  # ASK: 0 → 0, 1 → amplitude

    # Step 3: Compute signal power
    signal_power = np.mean(signal ** 2)

    # Step 4: Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10)

    # Step 5: Compute noise power
    noise_power = signal_power / snr_linear

    # Step 6: Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), num_bits)

    # Step 7: Add noise to signal
    noisy_signal = signal + noise

    return bits, signal, noisy_signal, noise

def main():
    n = get_m_bit_amount(M)
    bits_to_send = modulate_bit(sequence, int(n))
    time, values = modulate_ask(M, bits_to_send)

    plt.plot(time, values)

    # Add labels and title
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('Plot of y versus x')
    plt.grid(True)
    # plt.legend()

    # Show the plot
    plt.show()
    pass

if __name__ == "__main__":
    main()