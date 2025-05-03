# analysis_TLS_statistics.py 
Processes multiple days of data directories and produces time-series and statistical plots for key qubit metrics.  Analyses include:

1. **Single-Shot Fidelity (SSF)** – extract fidelity, rotation angles, thresholds; plot vs. time and histograms  
2. **Resonator Spectroscopy (Res Spec)** – track resonator frequencies; time-series, histograms, cumulative distributions, spectral densities, and Allan deviation  
3. **Qubit Spectroscopy (QSpec)** – track qubit transition frequencies with fit errors; time-series, histograms, error vs. value, spectral and Allan analyses  
4. **T₁ Relaxation (T1)** – track energy-relaxation times; time-series, histograms, cumulative distributions, spectral densities, and Allan deviation  

# analysis_comprehensive_TLS_v0.py
Loads and processes threshold, QSpec (low & high gain), SSF, T₁, resonator-Stark and detuning-Stark data from a given dataset. Analyses include:

1. ...