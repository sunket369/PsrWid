# **PsrWid**  
## **Master's Thesis Project**  

Pulsars are highly magnetized, rapidly rotating neutron stars that emit electromagnetic radiation through their magnetic poles. These poles are misaligned with the rotational axis, creating a lighthouse effect as the star spins. An observer detects pulsar emissions when the radiation beam crosses their line of sight. The structure of this emission beam is closely linked to the geometry of the pulsar's magnetosphere. Therefore, accurately characterizing the pulse profile is crucial, as it provides insights into the underlying magnetospheric processes.  

This project focuses on detecting and quantifying the emission widths within a pulsar’s pulse profile while automating the process for efficiency. In addition to identifying the main emission components, the algorithm is designed to detect weak emission features in the off-pulse region, if present.  

### **About the PsrWid Algorithm**  

The **Python script `PsrWid.py`** contains the class `PsrWid()`, which provides a suite of functions for a detailed analysis of pulsar emission widths. The key features include:  

- **Gaussian Process Regression (GPR):** Fits the integrated pulse profile data to model its structure.  
- **Threshold Detection using the Kneedle Algorithm:** Determines the optimal threshold for identifying the on-pulse region.  
- **Width Estimation with Uncertainty:** Computes the pulse width along with associated errors.  
- **Calculation of Standard Width Metrics:** Supports the extraction of **\( W_{10} \)** and **\( W_{50} \)**, commonly used in pulsar studies.  
- **Detection of Weak Emission Components:** Identifies low-intensity emissions beyond the primary pulse.  
- **Visualization Tools:** Provides plots to interpret results, including standard pulse profiles, log-polar representations, and threshold detection curves.  

### **Usage Instructions**  

The **`PsrWid_Usage`** file contains detailed guidelines on how to apply the algorithm to integrated pulse profiles. It includes step-by-step instructions on:  

- Running the `PsrWid()` class to analyze pulsar widths.  
- Extracting key width metrics and uncertainties.  
- Interpreting output results with visualizations.  

For further details, refer to the documentation within the script. Contributions, suggestions, and improvements are welcome!  
