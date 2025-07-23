import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox, Button, Slider
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

    return math.pow(m, 2) - 1 <= SNR
    

def snr_interval(m):
    """Retourne l'intervalle de SNR permis pour la modulation M-aire."""

    if m <= 0:
        return (0, 0)
    min_snr = math.pow(m, 2) - 1
    max_snr = min_snr + 20
    return (min_snr, max_snr)


AMP_PORTEUSE = 1 # amp max de la porteuse
POINTS_PER_PERIOD = 200

def ask_modulate(symbols, M, fc, snr=0):
    """Modulation ASK (Amplitude Shift Keying)"""
    period = 1 / fc
    gap = AMP_PORTEUSE / M
    amp_lookup_table = np.linspace(gap, AMP_PORTEUSE, num=M, endpoint=True)
    amp_values = amp_lookup_table[symbols]
    sinValues = np.zeros(len(symbols) * POINTS_PER_PERIOD)
    timeValues = np.linspace(0, len(symbols) * period, num=POINTS_PER_PERIOD * len(symbols), endpoint=True)
    for i in range(len(symbols)):
        index1 = i * POINTS_PER_PERIOD
        index2 = index1 + POINTS_PER_PERIOD - 1
        sinValues[index1:index2] = amp_values[i] * np.sin(2 * math.pi * fc * timeValues[index1:index2])

    if snr > 0:
        noise = np.random.normal(0, 10**(-snr/20), size=sinValues.shape)
        sinValues += noise

    vline_time_index = np.arange(0, len(sinValues), POINTS_PER_PERIOD)
    vlines = timeValues[vline_time_index]

    return timeValues, sinValues, vlines

def fsk_modulate(symbols, M, fc, snr=0, bandwidth=1000):
    """Modulation FSK (Frequency Shift Keying)"""
    period = 1 / fc
    freq_lookup_table = np.linspace(fc, fc + bandwidth, num=M, endpoint=True)
    freq_values = freq_lookup_table[symbols]
    sinValues = np.zeros(len(symbols) * POINTS_PER_PERIOD)
    timeValues = np.linspace(0, len(symbols) * period, num=POINTS_PER_PERIOD * len(symbols), endpoint=True)
    for i in range(len(symbols)):
        index1 = i * POINTS_PER_PERIOD
        index2 = index1 + POINTS_PER_PERIOD - 1
        sinValues[index1:index2] = AMP_PORTEUSE * np.sin(2 * math.pi * freq_values[i] * timeValues[index1:index2])

    if snr > 0:
        noise = np.random.normal(0, 10**(-snr/20), size=sinValues.shape)
        sinValues += noise

    vline_time_index = np.arange(0, len(sinValues), POINTS_PER_PERIOD)
    vlines = timeValues[vline_time_index]

    return timeValues, sinValues, vlines

# Modulation PSK (Phase Shift Keying)
def psk_modulate(symbols, M, fc, snr=0):
    """Modulation PSK (Phase Shift Keying)"""
    period = 1 / fc
    phase_end =  np.pi * 2 - np.pi * 2 / M
    phase_lookup_table = np.linspace(0, phase_end , num=M, endpoint=True)
    phase_values = phase_lookup_table[symbols]
    sinValues = np.zeros(len(symbols) * POINTS_PER_PERIOD)
    timeValues = np.linspace(0, len(symbols) * period, num=POINTS_PER_PERIOD * len(symbols), endpoint=True)
    for i in range(len(symbols)):
        index1 = i * POINTS_PER_PERIOD
        index2 = index1 + POINTS_PER_PERIOD - 1
        sinValues[index1:index2] = AMP_PORTEUSE * np.sin(2 * math.pi * fc * timeValues[index1:index2] + phase_values [i])
    if snr > 0:
        noise = np.random.normal(0, 10 ** (-snr / 20), size=sinValues.shape)
        sinValues += noise

    vline_time_index = np.arange(0, len(sinValues), POINTS_PER_PERIOD)
    vlines = timeValues[vline_time_index]

    return timeValues, sinValues, vlines

def bit_to_symbol(bin_seq, n):
    """Convertit une séquence binaire en symboles avec n bits par symbole."""
    symbols = []
    for i in range(0, len(bin_seq), n):
        symbol = bin_seq[i:i+n]
        symbols.append(int(symbol, 2))
    return symbols

def main():
    """Fonction principale pour l'interface graphique de modulation de signaux."""

    # Interface graphique avec matplotlib
    fig = plt.figure(figsize=(10, 5))
    plot_ax = fig.add_axes([0.05, 0.15, 0.65, 0.75])
    plot_ax.set_title("Signal modulé")
    plot_ax.set_xlabel("Temps (s)")
    plot_ax.set_ylabel("Amplitude")
    plot_ax.grid(True)

    # Textbox pour la séquence binaire
    fig.text(0.875, 0.85, 'Séquence binaire', ha='center', va='bottom', fontsize=10)
    ax_bin_seq = plt.axes([0.80, 0.8, 0.15, 0.05])
    text_bin_seq = TextBox(ax_bin_seq, "", initial='00100111')

    # Radio buttons pour la modulation
    fig.text(0.875, 0.75, 'Modulation', ha='center', va='bottom', fontsize=10)
    ax_radio = plt.axes([0.80, 0.65, 0.15, 0.10])
    radio_mod = RadioButtons(ax_radio, ('ASK', 'PSK', 'FSK'))

    # Textbox pour l'ordre M
    fig.text(0.875, 0.60, 'Ordre M', ha='center', va='bottom', fontsize=10)
    ax_m = plt.axes([0.80, 0.55, 0.15, 0.05])
    text_m = TextBox(ax_m, '', initial='4')

    # Textbox de la bande passante
    fig.text(0.875, 0.50, 'Bande passante (Hz)', ha='center', va='bottom', fontsize=10)
    ax_bandwidth = plt.axes([0.80, 0.45, 0.15, 0.05])
    text_bandwidth = TextBox(ax_bandwidth, '', initial='20000')

    # Textbox pour le SNR
    fig.text(0.875, 0.40, 'SNR (dB)', ha='center', va='bottom', fontsize=10)
    ax_snr_slider = plt.axes([0.80, 0.35, 0.15, 0.05])
    snr_slider = Slider(ax_snr_slider, '', snr_interval(4)[0], snr_interval(4)[1], valinit=0, valstep=0.1)

    # Message d'erreur:
    error_text = fig.text(0.875, 0.15, '', ha='center', va='top', fontsize=10, color='red', wrap=True)

    # Calcul de la capacité de Shannon-Hartley
    def shannon_capacity(bandwidth, snr):
        """Calcule la capacité de Shannon-Hartley en bits/s."""
        if bandwidth <= 0 or snr <= 0:
            return 0
        return bandwidth * np.log2(1 + 10**(snr / 10))

    # Afficher la capacité de Shannon-Hartley
    shannon_text = fig.text(0.875, 0.35, 'Capacité de Shannon-Hartley: - bits/s', ha='center', va='top', fontsize=10, wrap=True)

    def update_snr_on_m_change(text):
        """Met à jour le slider SNR en fonction de la valeur de M."""
        nonlocal snr_slider  # Permet de modifier la variable snr_slider dans la portée externe
        try:
            # Ne traiter que si le texte n'est pas vide
            if text.strip() == "":
                return
            
            m = int(text)
            if m <= 0:
                return
                
            min_snr, max_snr = snr_interval(m)
            
            # Sauvegarder la valeur actuelle
            current_val = snr_slider.val
            
            # Mettre la valeur dans les nouvelles limites
            new_val = max(min_snr, min(max_snr, current_val))
            
            # Recréer le slider avec les nouvelles limites
            ax_snr_slider.clear()
            snr_slider = Slider(ax_snr_slider, '', min_snr, max_snr, valinit=new_val, valstep=0.1)
            
            # Reconnecter l'événement de changement de valeur
            snr_slider.on_changed(lambda val: on_submit(None))
            
            # Redessiner
            fig.canvas.draw_idle()
            
        except ValueError:
            # Ne pas afficher d'erreur si l'utilisateur est en train de taper
            pass

    
    def on_submit(event):
        """Au clic du bouton, exécute la modulation et affiche le signal."""

        # Assignation des valeurs récupérées de l'interface graphique
        bin_seq = text_bin_seq.text                     # Séquence binaire
        modulation = radio_mod.value_selected           # Type de modulation
        m = int(text_m.text)                            # Ordre de la modulation M-aire
        SNR = float(snr_slider.val)                     # Rapport signal sur bruit en dB

        bandwidth = float(text_bandwidth.text)          # Bande passante en Hz

        # Validation de la séquence binaire
        if not all(bit in '01' for bit in bin_seq):
            error_text.set_text("La séquence binaire n'est pas valide. Elle doit contenir uniquement des 0 et des 1.")
            return
        
        # Validation de la modulation
        valid_modulations = ["ASK", "FSK", "PSK"]
        if modulation not in valid_modulations:
            error_text.set_text(f"La modulation '{modulation}' n'est pas supportée. Modulations supportées: {valid_modulations}")
            return
        
        # Validation du M
        is_power, m_sugg = is_power_of_two(m)
        if not is_power:
            error_text.set_text(f"Le paramètre M n'est pas une puissance de 2. M suggéré: {m_sugg}")
            m = m_sugg
            return

        # Calcul du nombre de bits par symbole
        n = math.log2(m)

        # Validation de la longueur de la séquence binaire
        if len(bin_seq) % n != 0:
            error_text.set_text(f"La longueur de la séquence binaire ({len(bin_seq)}) n'est pas un multiple de {n}.")
            return
        
        # Validation du SNR
        if not is_valid_snr(SNR, m):
            print(f"La longueur de la séquence binaire ({len(bin_seq)}) n'est pas un multiple de {n}.")
            return

        # Afficher le signal
        if modulation == "ASK":
            t, signal, vlines = ask_modulate(symbols=bit_to_symbol(bin_seq, int(math.log2(m))), M=m, fc=1000, snr=SNR)
        elif modulation == "FSK":
            t, signal, vlines = fsk_modulate(symbols=bit_to_symbol(bin_seq, int(math.log2(m))), M=m, fc=1000, snr=SNR, bandwidth=bandwidth)
        elif modulation == "PSK":
            t, signal, vlines = psk_modulate(symbols=bit_to_symbol(bin_seq, int(math.log2(m))), M=m, fc=1000, snr=SNR)
        else:
            print(f"Modulation '{modulation}' non supportée.")
            return

        # Affichage du signal avec matplotlib
        plot_ax.clear()
        plot_ax.plot(t, signal, label=f"Signal {modulation} (M={m})")
        plot_ax.vlines(vlines, ymin= -AMP_PORTEUSE, ymax= AMP_PORTEUSE, color='r', linestyle='dashed')
        plot_ax.set_title(f"Signal {modulation} pour la séquence binaire")
        plot_ax.set_xlabel("Temps (s)")
        plot_ax.set_ylabel("Amplitude")
        plot_ax.grid()
        plot_ax.legend()
        
        # Mettre à jour la capacité de Shannon-Hartley
        capacity = shannon_capacity(bandwidth, SNR)
        shannon_text.set_text(f'Capacité de Shannon-Hartley: {capacity:.2f} bits/s')
        
        fig.canvas.draw_idle()  # Redessiner le graphique
        error_text.set_text("")

    # Bouton
    button = Button(plt.axes([0.82, 0.2, 0.1, 0.05]), 'Exécuter')
    button.on_clicked(on_submit)
    
    # Afficher le graphique initial
    plot_ax.clear()
    plot_ax.set_title("Signal modulé")
    plot_ax.set_xlabel("Temps (s)")
    plot_ax.set_ylabel("Amplitude")
    plot_ax.grid()

    # Événements pour mettre à jour le graphique en temps réel
    text_bin_seq.on_submit(on_submit)                                       # Soumettre la séquence binaire
    radio_mod.on_clicked(lambda label: on_submit(None))                     # Changer de modulation
    text_m.on_submit(on_submit)                                             # Soumettre l'ordre M
    text_m.on_text_change(lambda text: update_snr_on_m_change(text))        # Mettre à jour le SNR en fonction de M
    snr_slider.on_changed(lambda val: on_submit(None))                      # Changer le SNR avec le slider

    # Afficher le graphique    
    plt.show()

if __name__ == "__main__":
    main()