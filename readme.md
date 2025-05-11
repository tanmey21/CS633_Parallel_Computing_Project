# Parallel Processing of 3D Time Series Volume Data

## Overview:
This project performs parallel processing of time series data for a 3D volume using C and MPI. The goal is to efficiently compute the following for every time step:

1. Count of local minima

2. Count of local maxima

3. Global minimum

4. Global maximum

The project uses 3D spatial domain decomposition to distribute the work across multiple MPI processes.

## Input Format:

The input file contains time series data of a 3D grid (nx × ny × nz) over multiple time steps. Data is arranged in XYZ order.

Example (if nx=2, ny=3, nz=2, nc=2):
Each row corresponds to a point (x,y,z), and each column to a time step.

## Usage:

### Compilaiton
mpicc -o executable src.c

### Run Command
mpirun -np <PX*PY*PZ> -f hostfile ./executable <input_file.txt> PX PY PZ NX NY NZ NC <output_file.txt>

## Output Format:
Output is written by rank 0 to the specified output file. The file contains:

Line 1: Pairs of integers for each timestep - (local minima count, local maxima count)

Line 2: Pairs of floats for each timestep - (global min, global max)

Line 3: Timings (in seconds) - Read Time, Main Code Time, Total Time

## Input Arguments:
| Argument Position | Description                               |
|-------------------|-------------------------------------------|
| 1                 | Input file name (.txt)                   |
| 2                 | PX - processes in X dimension            |
| 3                 | PY - processes in Y dimension            |
| 4                 | PZ - processes in Z dimension            |
| 5                 | NX - grid points in X dimension          |
| 6                 | NY - grid points in Y dimension          |
| 7                 | NZ - grid points in Z dimension          |
| 8                 | NC - number of time steps                |
| 9                 | Output file name                         |

## Results:
We are getting these results for an input size of 1e9 rows and 1 column.

### Table 3: Timing Results (seconds) for Larger Dataset

| Processes (np) | I/O & Distribution (s) | Computation (s) | Total (s) |
|----------------|-------------------------|------------------|-----------|
| 8              | 5.1426                  | 7.8324           | 12.2919   |
| 16             | 4.7381                  | 4.6730           | 9.4111    |
| 32             | 5.4094                  | 3.1695           | 8.5789    |
| 64             | 5.6933                  | 1.7483           | 7.4416    |
