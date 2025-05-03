# round_robin_benchmark.py
A round-robin coherence benchmarking script for superconducting qubits.  
Cycles through selected qubits in 'round robin' style, logging shifts and dynamics in various metrics over time. Optional measurements include:

1. **Time-of-Flight (TOF)** – measure signal propagation delay in the readout chain  
2. **Resonator Spectroscopy (Res Spec)** – sweep each readout resonator to locate its resonance peak  
3. **Qubit Spectroscopy (QSpec)** – sweep qubit drive frequency to find the qubit transition frequency  
4. **Single-Shot Readout (SS)** – calibrate & threshold single-shot I/Q histograms for state discrimination  
5. **Amplitude Rabi** – drive Rabi oscillations to extract the $\pi$-pulse amplitude  
6. **$T_1$ Measurement** – measure qubit energy-relaxation time ($T_1$)  
7. **$T_2$ Ramsey (T2R)** – measure dephasing time via Ramsey fringes  
8. **$T_2$ Echo (T2E)** – measure coherence using an echo sequence 
