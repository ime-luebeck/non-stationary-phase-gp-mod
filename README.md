# non-stationary-phase-GP-mod
This repository implements methods for highly scalable GP inference when kernels have a non-stationary phase. The corresponding paper is available on arXiv https://arxiv.org/abs/1912.11713.

The mod adds some new functionalities on top of the GPML (http://www.gaussianprocess.org/gpml/code/matlab/doc/). Basically it enables to use non-equidistant inducing point sets using a modified versions of the apx/apxGrid methods from GPML. Also, a new function is provided that allows to use mixtures of non-stationary kernels.

The user can define kernels in standard GPML manner. Right now, the mod only works with the Lanczos mode for stochastic trace estimation.

The code was tested with MATLAB 2019a.

## Example
Setup two kernels with a non-stationary phase (where `tau1` and `tau2` define the period length, i.e. between two elements in `tau`, a phase difference of $`2\pi`$ is assumed):

```Matlab
cov_per_1 = @covPeriodic;
p1 = @(t) (timeWrap( t, tau1 ));
dp1 = @(t) (dtimeWrap( t, tau1));
covPeWarped_1 = {@covWarp,{cov_per_1},p1,dp1,1};

cov_per_2 = @covPeriodic;
p2 = @(t) (timeWrap( t, tau2 ));
dp2 = @(t) (dtimeWrap( t, tau2));
covPeWarped_2 = {@covWarp,{cov_per_2},p2,dp2,1};
```

Then construct some non-equidistant inducing points:
```Matlab
U = (0:0.05:10, tau1))';
U_hat_1 = invTimeWrap( U, tau1);
U= (0:0.05:20, tau2))';
U hat_2 = invTimeWrap( U, tau2);

xg_1 = {{U_1}};
xg_2 = {{U_2}};

```

Then build the full kernel as the sum of the two non-stationary kernels:

```
csu_grid = {'apxGridSum',{{@apxGrid, {covPeWarped_1}, xg_1},{@apxGrid, {covPeWarped_2}, xg_2}}};
```
You can now use this kernel for standard GPML inference, e.g. you can pass it to the `infGaussLik` function.

## Installation
1. Download GPML v4.2 from http://www.gaussianprocess.org/gpml/code/matlab/doc/ and extract it into the directory where `startup.m` is located.
2. Run `startup.m` to load all necessary files. The GPML will be loaded and the modified files will be loaded to a higher position in the MATLAB search path.

3. As an initial test for the provided functionality you can run `demo_non_stationary_1D_gridbased.m`.

4. Installation of the L-BFGS optimizer: compile the FORTRAN code provided with GPML – instructions on how to do so can be found here: http://www.gaussianprocess.org/gpml/code/matlab/doc/README
You might also consider to just use `minimize()` instead of `minimize_lbfgsb()` but the performance of the latter is significantly higher – in particular when the gradients are calculated by stochastic estimators.

## Running the numerical experiment
To generate the numerical example from the paper:
1.	Run `demo_non_stationary_2D_grid_based_stresstest.m`.
2.	To generate the plots from the paper, run `results/results_stresstest/plots_results_stresstest.m`.

## Running the fetal ECG experiment
1.	Download `r01.edf` from https://physionet.org/content/adfecgdb/1.0.0/ and store the data to the `/demo_fetal_ecg/` directory.
2.	Download the EDF reader from https://de.mathworks.com/matlabcentral/fileexchange/31900-edfread, unzip the file and add it to your MATLAB path.
3.	Run `fetal_ecg.m`.
4.	To generate the plots from the paper, run `results/results_fetal_ecg/plot_results_fetal_ecg.m`.

## Running the EIT experiment
1.	Download eidors-v3.9.1 from http://eidors3d.sourceforge.net/download.shtml and extract all files into the `demos/demo_EIT/` directory.
2.	Download ready made FEM models from https://iweb.dl.sourceforge.net/project/eidors3d/eidors-v3/eidors-v3.6/model_library.zip and extract the files into the directory `/eidors-v3.9.1/eidors/models/cache`.
3.	Download data from http://eidors3d.sourceforge.net/data_contrib/if-neonate-spontaneous/index.shtml and copy the data into the `demos/demo_EIT` directory
4.	Run `eit_load_data.m` to solve the EIT reconstruction. A mat file containing the EIT images is generated and stored into the `demos/demo_EIT/if-neonate-spontaneous` directory.
5.	The file `eit_demo.m` solves the perfusion and ventilation separation with fixed hyperparameters.
6.	The file `eit_demo_with_learning.m` solves the perfusion and ventilation separation and optimizes the hyperparameters. Careful: the optimization takes a long time (around 9 to 10 hours).
7.	To generate the plots from the paper, either run `results/results_eit/v1/plot_results_eit_v1.m` or `results/results_eit/v1/plot_results_eit_v2.m`

