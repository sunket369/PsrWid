# PsrWid
## Master's Thesis Project

Pulsars are highly magnetized and rapidly rotating neutron stars. They emit electromagnetic radiation through their poles which are aligned with some angle w.r.t. the rotating axis - creating a lighthouse effect. The observer ‘sees’ these pulsars when the radiation beam cuts off the observer’s line of sight. The structure of this beam is correlated with the geometry of the magnetosphere. Hence it is important to characterize the pulse profile, which could indirectly help to study the magnetosphere. The Project aims to detect the widths of emission in the pulse profile of a Pulsar and automate the process. This extends to finding the weak emission from the off-pulse region if it presents.

Python script `PsrWid.py` contains the class `PsrWid()` with different functions that can be used to perform a detailed analysis of pulsar widths. It includes fitting the integrated pulse profile data with the Gaussian Process and determining the width and its error with the 'Kneedle Algorithm'. It also has the option to get $W_{10}$ and $W_{50}$.
