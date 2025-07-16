import numpy as np
import matplotlib.pyplot as plt
import math

def is_power_of_two(m):
    if m <= 0:
        return False, 0
    elif math.log2(m).is_integer():
        m_sugg = nearest_power_of_two(m)
        return True, m_sugg
    else:
        m_sugg = nearest_power_of_two(m)
        return False, m_sugg
    
def nearest_power_of_two(m):
    n = int(round(math.log2(float(m))))
    return 2 ** n  

def main():
    
    bin_seq = "10110010"        # Sequence binaire
    modulation = "ASK"          # Type de modulation (ASK, FSK, PSK)
    m = 2                       # Ordre de la modulation M-aire
    SNR = 1                     # Rapport signal sur bruit en dB
    bandwidth = 10              # Bande passante en Hz
    n = 2                       # Nombre de bit par symbole

    fs = 1000                   # Fréquence d'échantillonnage
    fc = bandwidth/2            # Fréquence porteuse




if __name__ == "__main__":
    main()