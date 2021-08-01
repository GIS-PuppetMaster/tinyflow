# We implemented vDNN, Capuchin and TENSILE on the modified Tinyflow for fair comparison.
Tinyflow is a simple deep learning framework for learning purposes. It supports automatic 
differentiation and GPU acceleration. The modified TinyFlow currently provides all the operators needed 
to build a multilayer perceptron models (MLP), and a convolution network.

If you want to learn more about the principles behind Tinyflow, the following two blog posts may provide a lot of intuition.
+ [Automatic Differentiation Based on Computation Graph](https://lb-yu.github.io/2019/07/22/Automatic-Differentiation-Based-on-Computation-Graph/)
+ [Tinyflow - A Simple Neural Network Framework](https://lb-yu.github.io/2019/07/23/Tinyflow-A-Simple-Neural-Network-Framework/)


# Install
Tinyflow currently only supports running in 64-bit linux environment. Requirement:
+ Ubuntu 18.04
+ gcc >= 4.8 (We used gcc 7.5);
+ cmake >= 3.13 (if you choose to use cmake);
+ CUDA 10.0
+ cudnn 7.6.5
+ python 3

Download the source code.
```shell
git clone https://github.com/GIS-PuppetMaster/tinyflow.git
```

Generally speaking, CUDA will be installed in `/use/local/cuda`. 
If your installation path is different, please modify the `CUDA_DIR` variable on the first 
line of the Makefile to your installation path, or modify the `CUDA_DIR` variable on the 
fourth line of CMakeLists.txt to your installation path.

For compiling with Makefile.
```shell
make
```

For compiling with CMake.
```shell
mkdir build
cmake ..
make
make install
```

# Run the experiments of TENSILE
After compiling the GPU library, we can check out to the TENSILE branch and run the experiments of TENSILE
```shell
# for single workload and multiple dynamic workloads experiments
python pycode/tinyflow/MainExperiments.py
# for various multiple dynamic workloads experiments
python pycode/tinyflow/VariousMultiDynamicWorkloadsExperiments.py
```
# Run the experiments of vDNN and Capuchin
Back up the results of TENSILE first.
Check out to the baselines branch first, and re-compile first, then manually change the budget in lab1.py and lab3.py to the results of your TENSILE experiment. 
```shell
# re-compile
rm -rf build
make
# for single workload and multiple dynamic workloads experiments
python tests/Experiment/lab1.py
# for various multiple dynamic workloads experiments
python tests/Experiment/lab2.py
```

# Get results
1. Back up the results of baselines.
2. Check out back to the TENSILE branch.
3. Change the `baseline_path` and `TENSILE_path` in `MakeCSV.py` to the path where you put your experiment results.
4. Change the `csv_path` in `Draw.py` to the path where you put your csv, this is set in MakeCSV.py.
5. Run the result analysis process.
```shell
python pycode/tinyflow/Draw.py
```
6. Check the picture and csv file in `csv_path`
