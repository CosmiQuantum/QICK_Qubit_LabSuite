## Fridge Configurations

Each fridge folder contains two JSON files—`measurement_config.json` and `system_config.json`—for running experiments on its dedicated qubit array.

**Current fridges:**
- **NEXUS**: Northwestern’s underground fridge at Fermilab  
- **QUIET**: QSC’s underground fridge at Fermilab  

---

## `measurement_config.json`

Per‐experiment parameters for QICK benchmarking:

- **tot_num_of_qubits**  
- **VNA_res** / **VNA_qubit**  
- **expt_cfg**:  
  1. **Time‐of‐Flight** (`tof`)  
  2. **Resonator Spectroscopy** (`res_spec`, `res_spec_ef`)  
  3. **Qubit Spectroscopy** (`qubit_spec_ge`, `qubit_spec_ef`, `bias_qubit_spec_ge`)  
  4. **Rabi Oscillations** (`power_rabi_ge`, `power_rabi_ef`)  
  5. **Decoherence** (`T1_ge`, `Ramsey_ge`, `SpinEcho_ge`, `Ramsey_ef`)  
  6. **Temperature Sweep** (`qubit_temp`)  
  7. **Readout Calibration** (`IQ_plot`, `Readout_Optimization`)  

---

## `system_config.json`

Global QICK settings for hardware control and measurement chain:

- DAC/ADC attenuations  
- Readout & pulse sequencing parameters  
- …additional device‐specific settings…  
