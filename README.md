# Part-III-ML-Centroiding-Project
## Intro
Some example programs from my master's project, 'Machine Learning to Centroid Speckled Images'.

Adaptive optics (AO) systems are used in astronomical imaging to correct for wavefront perturbations of light created by Earth’s atmosphere. The lowest order AO systems are called ‘tip/tilt’ systems which aim to remove the linear component of the phase by estimating the image centroid. High order aberrations in the phase cause ‘speckling’ of the image which leads to a centroiding error in most commonly used methods.  

This project uses convolutional neural networks to better estimate image centroids. Networks are trained and tested using a computational simulation of atmospheric perturbations which models a Von Kàrmàn spectrum. The context for the investigation is the Magdalena Ridge Optical Interferometer (MROI) being built in New Mexico. 

The project tested the networks over a range of factors important for the MROI telescope these included the effect of undersampling data, effects of varying turbulence (D/r0) value, and of shot noise. 

## Code
Included in the repository are two of the fundemental scripts. The first is used to generate data for training/testing. This involves using a Von-Karman spectrum to simulate atmospheric turbulence. A diffraction image is then created using a FFT and a set of Zernikes is generated and the spatial component in each mode is measured to get the tip/tilt.  
The second is the network trainer which uses tensorflow (with keras) to train different neural network architectures.  
