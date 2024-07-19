# K-Means Clustering Implementation

This project is an implementation of the K-Means clustering algorithm that can be executed on both the host CPU and the GPU using CUDA. It includes different optimization levels for the CPU execution and provides flexibility in running the algorithm on various datasets.

## Requirements

- GCC (GNU Compiler Collection)
- NVCC (NVIDIA CUDA Compiler)
- OpenMP (optional for parallel processing on CPU)

## Project Structure

- `cluster.c`, `getopt.c`, `kmeans.c`, `kmeans_clustering.c`: Source files for CPU implementation.
- `kmeans_clustering.cu`: Source file for GPU implementation using CUDA.
- `kmeans.h`: Header file for K-Means clustering.
- `kmeans_data/`: Directory containing datasets for K-Means clustering.

## Makefile Targets

To change the dataset, modify the DATA_DIR and FILE_NAME variables in the Makefile:

### Dataset selection
DATA_DIR = <your_datasets_directory>
FILE_NAME = <your_dataset_file>

### Compilation

- `make compile_all`: Compiles all versions of the K-Means clustering implementation.
- `make compile_host`: Compiles the CPU version without vectorization.
- `make compile_host_O3`: Compiles the CPU version with vectorization.
- `make compile_gpu`: Compiles the GPU version using CUDA.

### Execution

- `make run_host`: Runs the CPU version without vectorization.
- `make run_host_O3`: Runs the CPU version with vectorization.
- `make run_gpu`: Runs the GPU version using CUDA.
- `make run_all`: Runs all versions sequentially and compares their performance.

### Cleaning

- `make clean`: Cleans up all object files and executables.

Usage: ./kmeans [switches] -i filename
       -i filename     	  :  file containing data to be clustered
       -b                 :  input file is in binary format
       -k                 :  number of clusters (default is 8) 
       -t threshold       :  threshold value
