import numpy as np
import matplotlib.pyplot as plt
import math


def is_power_of_two(m):
    """Vérifie si m est une puissance de 2 et retourne la puissance de 2 la plus proche si ce n'est pas le cas."""

    if m <= 0:
        return False, 0
    elif math.log2(m).is_integer():
        m_sugg = nearest_power_of_two(m)
        return True, m_sugg
    else:
        m_sugg = nearest_power_of_two(m)
        return False, m_sugg
    
def nearest_power_of_two(m):
    """Retourne la puissance de 2 la plus proche de m."""

    n = int(round(math.log2(float(m))))
    return 2 ** n  

def is_valid_snr(SNR, m):
    """Vérifie si le SNR est suffisant pour la modulation M-aire."""

    if math.pow(2, m) - 1 <= SNR:
        return True
    else:
        return False
    

def main():
    
    bin_seq = "10110010"        # Sequence binaire
    modulation = "ASK"          # Type de modulation (ASK, FSK, PSK)
    m = 2                       # Ordre de la modulation M-aire
    SNR = 1                     # Rapport signal sur bruit en dB
    bandwidth = 10              # Bande passante en Hz
    fs = 1000                   # Fréquence d'échantillonnage
    fc = bandwidth/2            # Fréquence porteuse

    # Validation de la séquence binaire
    if not all(bit in '01' for bit in bin_seq):
        print("La séquence binaire n'est pas valide. Elle doit contenir uniquement des 0 et des 1.")
        return
    
    # Validation de la modulation
    valid_modulations = ["ASK", "FSK", "PSK"]
    if modulation not in valid_modulations:
        print(f"La modulation '{modulation}' n'est pas supportée. Modulations supportées: {valid_modulations}")
        return
    
    # Validation du M
    is_power, m_sugg = is_power_of_two(m)
    if not is_power:
        print(f"Le paramètre M n'est pas une puissance de 2. M suggéré: {m_sugg}")
        m = m_sugg

    # Calcul du nombre de bits par symbole
    n = math.log2(m)

    # Validation de la longueur de la séquence binaire
    if len(bin_seq) % n != 0:
        print(f"La longueur de la séquence binaire ({len(bin_seq)}) n'est pas un multiple de {n}.")
        return

    # Validation du SNR
    if not is_valid_snr(SNR, m):
        print(f"Le SNR n'est pas suffisant pour la modulation M-aire avec M={m}.")
        return
    







    # Affichage du signal avec matplotlib
    t = np.arange(0, len(bin_seq) * n / fs, 1/fs)
    signal = np.zeros(len(t))




if __name__ == "__main__":
    main()