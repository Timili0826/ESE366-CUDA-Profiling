# CUDA Profiling on Jetson Orin Nano

ESE 366, Stony Brook University  
**Author:** Tim Li

This project uses CUDA events, compiler resource reports, Nsight Systems, Nsight Compute, and `tegrastats` to investigate CUDA performance. The experiments compare the cost of vector addition with data movement, then examine how read stride affects memory-access efficiency.

## Repository contents

- [`Codes/`](Codes/): CUDA source files.
  - [`device_query.cu`](Codes/device_query.cu): reports GPU properties and available memory.
  - [`vector_add.cu`](Codes/vector_add.cu): performs vector addition and measures a CUDA event interval.
  - [`memory_stride.cu`](Codes/memory_stride.cu): compares read strides 1 and 32, with a warm-up for each case.
- [`Results/`](Results/): captured compiler, timing, profiling, and monitoring outputs.

## Experimental setup

| Property | Configuration |
|---|---|
| Platform | NVIDIA Jetson Orin Nano Developer Kit Super |
| Compute capability | 8.7 |
| SMs / warp size | 8 / 32 threads |
| Power mode | 25 W |
| Fixed CPU / GPU clock | 1.344 GHz / 918 MHz |
| Elements per experiment | 1,048,576 |
| Launch configuration | 4096 blocks, 256 threads per block |

## Main findings

### Data movement outweighed vector addition

Nsight Systems recorded the following GPU operation durations:

| Operation | Total time |
|---|---:|
| Two host-to-device copies | 0.985312 ms |
| Vector-addition kernel | 0.177664 ms |
| Device-to-host copy | 2.057824 ms |

Copies accounted for **94.5% of the summed GPU copy and kernel durations**, not total application wall time. The first `cudaMalloc` also took approximately 75.4 ms on the CPU, consistent with substantial first-use initialization cost. For repeated computations, these observations suggest investigating allocation reuse and keeping intermediate data on the device.

### Access pattern mattered more than occupancy alone

| Measurement | Stride 1 | Stride 32 |
|---|---:|---:|
| Standalone time after warm-up | 0.134176 ms | 0.589664 ms |
| Achieved occupancy | 81.07% | 89.92% |
| Global-load requests | 32,768 | 32,768 |
| Global-load sectors | 131,072 | 1,048,576 |
| Sectors per request | 4 | 32 |

Stride 32 was **4.39 times slower**, despite higher achieved occupancy. Scattered reads required eight times as many L1/TEX load sectors per request for the same useful element count. These counters describe load-sector activity, not direct DRAM traffic.

The main lesson was to locate the expensive stage, use profiling evidence to investigate its cause, and then measure whether an optimization helps.

## Connection to my first course project

My first course project, [ESE366-CUDA](https://github.com/Timili0826/ESE366-CUDA), explored CUDA implementations and optimizations for matrix operations. This profiling lab gives me a method for investigating why those changes affect performance.

| Earlier optimization topic | What profiling can help investigate |
|---|---|
| Matrix addition and loop unrolling | Compare runtime and register demand to assess the tradeoff between work per thread and resource usage. |
| Matrix transpose | Use load/store sector counters to examine coalescing; use shared-memory bank-conflict counters to evaluate padding. |
| Tiled matrix multiplication | Examine whether data reuse reduces global-memory traffic and how shared-memory usage affects occupancy. |
| Trace reduction | Compare reduction variants using execution time, memory-access behavior, and synchronization costs. |

The connection is a change in how I evaluate an optimization: measure its effect, identify a possible explanation, and check that explanation with profiling evidence. These are directions for analyzing the first project, not additional measurements of its kernels.

## Build and run

Run on the Jetson with the CUDA toolkit installed. From the repository root:

```bash
cd Codes
nvcc -arch=sm_87 device_query.cu -o device_query
nvcc -arch=sm_87 --ptxas-options=-v vector_add.cu -o vector_add
nvcc -arch=sm_87 --ptxas-options=-v memory_stride.cu -o memory_stride

./device_query
./vector_add
./memory_stride
```

Expected spot checks are `c[100] = 300.000000` and `b[100] = 1.0`. Exact timings depend on operating conditions.

## Profiling workflow

Check the power mode and fix clocks before collecting comparable measurements:

```bash
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show
```

From `Codes/`, collect application and kernel profiles:

```bash
nsys profile --stats=true -o vector_add_profile ./vector_add
ncu --set basic ./memory_stride
```

Hardware counters may require elevated permissions. If `ncu` works in the normal shell but `sudo ncu` cannot find it, resolve its path first:

```bash
command -v ncu
# If a path was returned:
sudo "$(command -v ncu)" --set basic ./memory_stride
```

Collect the global-load counters used in the stride comparison:

```bash
sudo "$(command -v ncu)" \
  --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  ./memory_stride
```

The stride program launches four kernels in order: stride-1 warm-up, stride-1 timed launch, stride-32 warm-up, and stride-32 timed launch. Compare the second and fourth entries for the timed cases. Open the generated `.nsys-rep` file with the Nsight Systems GUI to inspect the timeline.

For runtime monitoring, run `tegrastats --interval 100` in another terminal. Individual kernels are shorter than the sampling interval, so repeated program runs make GPU activity easier to observe.

## Measurement notes

- Standalone stride timings include a warm-up; vector-addition timing does not.
- CUDA events measure an elapsed interval, which can include waiting around the first kernel launch. Nsight Systems recorded a shorter kernel duration than the event interval in this experiment.
- Nsight Compute instrumentation and replay affect application event timing. Use standalone runs for the reported runtime comparison.
- Timings are individual measurements; correctness checks cover selected elements rather than the full output arrays.
- Jetson has integrated memory, but explicit copies in these programs still have measurable cost.
- `tegrastats` GPU utilization is different from occupancy, and device input power is not GPU-only power.

## References

- Alex Doboli, *CUDA Profiling*, ESE 366/ESE 566 laboratory slides, Stony Brook University.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
