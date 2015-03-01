## DpMM
A set of Dirichlet Process Mixture Model (DPMM) sampling-based inference algorithms.

This is research code and builds on the following two papers (please cite them appropriately):
- [1] Jason Chang and John W. Fisher III.
  Parallel Sampling of DP Mixture Models using Sub-Clusters Splits,
  NIPS 2013.
- [2] Julian Straub, Jason Chang, Oren Freifeld and John W. Fisher III.
  Dirichlet Process Mixture Model for Spherical Data,
  AISTATS 2015.
 
### Dependencies

This code depends on the following other libraries: Eigen3, Boost, CUDA. Optionally OpenMP. 

It has been tested under Ubuntu 14.04 with: 
- Eigen3 (3.0.5) 
- Boost (1.52)
- CUDA (6.5)

It has been tested under Windows 7 with: 
- Visual Studio 2012
- Eigen3 (3.2.3) 
- Boost (1.57)
- CUDA (6.5)

It has been tested under Mac OS (10.9.4) with: 
- clang-600.51 
- Eigen3 (3.2.1) 
- Boost (1.55)
- CUDA (6.5)
- No OpenMP


### Getting Started
Get started quickly by looking at `python/dpmmSampler.py`. It simply loads a
dataset and runs the sub-cluster split/merge algorithm with different base
measures:
```
python ./python/dpmmSampler.py -i ./data/rndSphereDataIwUncertain.csv -b DpNiwSphereFull -T 400
python ./python/dpmmSampler.py -i ./data/rndSphereDataIwUncertain.csv -b DpNiw -T 400
```
Where DpNiwSphereFull is for the DP-TGMM [2] and DpNiw for the standard DP-GMM
[1]. 
After finishing the specified number of iterations (via the `-T` option) the log
likelihood as well as the number of clusters over the iterations is shown.
Make sure you compiled the cpp code beforehand since the python script
just wraps the call to dpmmSampler.

### Executables
- *dpmmSampler*: Sampler for Dirichlet process mixture model (DPMM) inference using different algorithms
```
Allowed options:
  -h [ --help ]         produce help message
  --seed arg            seed for random number generator
  -N [ --N ] arg        number of input datapoints
  -D [ --D ] arg        number of dimensions of the data
  -T [ --T ] arg        number of iterations
  -a [ --alpha ] arg    alpha parameter of the DP (if single value assumes all 
                        alpha_i are the same
  -K [ --K ] arg        number of initial clusters 
  --base arg            which base measure to use (NIW, DpNiw, DpNiwSphereFull,
                          DpNiwSphere, NiwSphere, NiwSphereUnifNoise, spkm, 
                        spkmKarcher, kmeans right now)
  -p [ --params ] arg   parameters of the base measure
  --brief arg           brief parameters of the base measure (ie Delta = 
                        delta*I; theta=t*ones(D)
  -i [ --input ] arg    path to input dataset .csv file (rows: dimensions; 
                        cols: different datapoints)
  -o [ --output ] arg   path to output labels .csv file (rows: time; cols: 
                        different datapoints)
```

### Collaborators
Julian Straub (jstraub) and Randi Cabezas (rcabezas)

