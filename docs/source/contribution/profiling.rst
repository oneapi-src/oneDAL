.. Copyright contributors to the oneDAL project
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Using oneDAL Verbose Mode
=========================

When building applications that call Intel® oneAPI Data Analytics Library functions, it may be useful to determine:

- Which computational kernels are called
- What parameters are passed to them
- How much time is spent to execute the functions
- Which device (CPU/GPU) the kernel is executed on

You can get an application to print this information to a standard output
device by enabling **Intel® oneAPI Data Analytics Library Verbose Mode**.
All logs are printed to the `stderr` stream.

When Verbose mode is active in oneDAL, every call of a verbose-enabled function finishes with 
printing a human-readable line describing the call. Even if the application gets terminated during 
the function call, information for that function will be printed. 
The first call to a verbose-enabled function also prints a version information line.
The analyzer becomes available only at the end of the oneDAL workflow, when static objects are destroyed.

For GPU applications, additional information (one or more GPU information lines) will also 
be printed on the first call to a verbose-enabled function, following the version information lines printed
for the host CPU.

Verbosity Levels
----------------

We have common implementations for verbose mode with CPU and GPU applications.

 Levels:
  - Disabled (default)
  - Logger enabled
  - Tracer enabled
  - Analyzer enabled
  - All tools enabled
  - All tools and service functions enabled

Setting Verbose Mode
--------------------

You can change the verbose mode either by setting the environment variable ``ONEDAL_VERBOSE``.

**Using environment variable:**

+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=0`` | Disabled (default)                            |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=1`` | Logger enabled                                |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=2`` | Tracer enabled                                |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=3`` | Analyzer enabled                              |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=4`` | All tools enabled (Logger + Tracer + Analyzer)|
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=5`` | All tools and service functions enabled       |
+--------------------------+-----------------------------------------------+


The analyzer and the timeline can also record memory and write a timeline file. These are set
with their own environment variables:

+-----------------------------------------+----------------------------------------------------------------+
| ``ONEDAL_PROFILER_TRACE=<file>``        | Write a timeline of every task to ``<file>`` at exit (see      |
|                                         | :ref:`profiling-timeline`). Enables profiling on its own.      |
+-----------------------------------------+----------------------------------------------------------------+
| ``ONEDAL_PROFILER_TRACE_LIMIT=<n>``     | Keep at most ``<n>`` timeline events (default 4000000).        |
+-----------------------------------------+----------------------------------------------------------------+
| ``ONEDAL_PROFILER_MEMORY=0``            | Do not record allocations.                                     |
+-----------------------------------------+----------------------------------------------------------------+
| ``ONEDAL_PROFILER_OUTPUT=<file>``       | Write the Logger, the Tracer and the report to ``<file>``      |
|                                         | instead of ``stderr``, with every call logged.                 |
+-----------------------------------------+----------------------------------------------------------------+
| ``ONEDAL_PROFILER_REPORT=tree``         | Print only the task tree: no version lines, summary, hotspots  |
|                                         | or memory summary.                                             |
+-----------------------------------------+----------------------------------------------------------------+

GPU tasks that are given a queue wait for the queue when they start and when they end, so their time
covers the device work. This serializes kernels that would otherwise overlap.

Logger
------

**Logger** is a subtool used to capture information about each launched kernel.

Example output (for ``cov_dense_batch``):

::

    auto oneapi::dal::covariance::backend::compute_kernel_dense_impl<float>::operator()(const descriptor_t &, const parameters_t &, const input_t &)::(anonymous class)::operator()() const [Float = float]
    Profiler task_name: allreduce_sums, q_

    auto oneapi::dal::covariance::backend::compute_kernel_dense_impl<float>::operator()(const descriptor_t &, const parameters_t &, const input_t &)::(anonymous class)::operator()() const [Float = float]
    Profiler task_name: gemm, q_

Each entry shows the ``__PRETTY_FUNCTION__`` along with the corresponding ``PROFILER_TASK`` and its arguments.

The primary purpose of the Logger is debugging — especially useful for identifying the last called function before a segmentation fault or crash (if the function includes a profiler tag).
For that, set ``ONEDAL_PROFILER_OUTPUT=<file>``: the file gets every call and is flushed after each line,
while ``stderr`` gets each call site only once.


Tracer
------

**Tracer** is a subtool designed to collect and display the execution time of each ``profiler_task``.

Example output (for ``cov_dense_batch``):

::

    Covariance::prepareSums 8.40 us
    Covariance::prepareCrossProduct 1.26 us
    THREADING computeSums.local finished on the main rank(time could be different for other ranks): 2.36 us
    THREADING computeSums.reduce finished on the main rank(time could be different for other ranks): 630 ns

The Tracer is particularly useful in Jupyter Notebooks or general profiling scenarios where performance insights are needed.


Analyzer
--------

**Analyzer** prints, at exit, the tree of every task the program ran, with time and memory for each.
Tasks with the same name under the same parent are merged into one row. Tasks that run on worker
threads are placed under the task that started their parallel region.

Example output (illustrative):

::

    oneDAL profiler report
      profiled wall time 120.00 ms on 32 threads, 12000 task runs
      profiler cost about 60 ns per task run, 720.00 us of CPU time in total (0.4% of the profiled CPU time)

          time      %      self    calls       avg       max     thr host alloc  host peak  task
     120.00 ms  100.0  10.00 ms       10  12.00 ms  14.00 ms       1  20.00 MiB   2.00 MiB  CovarianceDenseBatchKernel::compute
      20.00 ms   16.7  20.00 ms       10   2.00 ms   2.50 ms       1        0 B        0 B  |-- Covariance::prepareCrossProduct
      90.00 ms   75.0   5.00 ms       10   9.00 ms  11.00 ms       1  20.00 MiB   2.00 MiB  `-- Covariance::updateDenseCrossProductAndSums
      85.00 ms   70.8   2.00 ms     2000  1.20 ms   3.00 ms      32        0 B        0 B      `-- reducer.update  [parallel, cpu 2.40 s, imbalance 1.30]
      83.00 ms   69.2  83.00 ms     2000  1.15 ms   2.90 ms      32        0 B        0 B          `-- reducer.update.syrkData  [parallel, cpu 2.30 s, imbalance 1.30]

    Hotspots by self time
          self      %    calls  task
      83.00 ms   69.2     2000  reducer.update.syrkData
      20.00 ms   16.7       10  Covariance::prepareCrossProduct
      ...

    Memory
      host:   allocated 300.00 MiB in 12000 blocks, freed 300.00 MiB in 12000 blocks, peak 6.00 MiB, live now 0 B

The columns are:

- ``time``: wall time. A task that runs in a parallel region is as long as its busiest thread in
  each run of the region, so a child is never longer than its parent.
- ``%``: share of the profiled wall time.
- ``self``: time not spent in child tasks, or waiting for parallel regions the task started.
- ``calls``, ``avg``, ``max``: number of runs, and the average and longest run.
- ``thr``: number of threads; for a parallel task, the most threads in one run of its region.
- ``host alloc``, ``dev alloc``: bytes allocated by the task and its children. Allocations that a
  worker makes outside any task count toward the task that started its parallel region.
- ``host peak``, ``dev peak``: the highest live memory above the level at the start of the task.
- ``to dev``, ``to host``: bytes copied between host and device memory.

A task that logs its arguments, such as ``blas.gemm`` with its matrix sizes, gets a row for each
distinct set of them, for example ``blas.gemm [a.get_dimension(0): 20; a.get_dimension(1): 1000; alpha: 1]``.
After eight such rows under one parent, further argument sets share a row named ``[other args]``.

Parallel tasks also show their total CPU time and their imbalance: the busiest thread over the
average thread in each run, where 1.00 is a perfectly balanced region. A row named
``(waiting for parallel work)`` is the time a task waited for a parallel region it started, beyond
running chunks of the region itself.

The memory columns, and the transfer columns, appear only when there is something to show: a CPU
run has no device or transfer columns, and no host columns either if every allocation happened
outside the profiled tasks. The ``Memory`` summary at the end always covers the whole process.
Allocations go through ``daal_malloc`` and the oneAPI USM allocator; memory allocated in any other
way is not seen.

.. _profiling-timeline:

Timeline
--------

``ONEDAL_PROFILER_TRACE=<file>`` writes every task run, per thread, and the live host and device memory
over time, in the Chrome trace event format. Open the file in https://ui.perfetto.dev or in
``chrome://tracing`` to see when each task ran on which thread, where threads were idle, and how memory
grew. When the number of events exceeds ``ONEDAL_PROFILER_TRACE_LIMIT``, later events are dropped and a
warning is printed.

General Notes on Verbose Mode
-----------------------------

The analyzer report and the timeline are written when the process exits. They are not written in
interactive environments like Jupyter Notebooks, where the process may not exit; run the application
as a standalone script.

On ``stderr``, the Logger and the Tracer print each call site once. Every call, which for an iterative
algorithm can be megabytes, goes only to the file named by ``ONEDAL_PROFILER_OUTPUT``: where ``stderr``
is a pipe that is not read while oneDAL runs, such as a notebook kernel, that much output would block
the process once the pipe is full.

Profiling adds the cost printed in the report to every task run. The report shows it so that the
timings can be judged against it.
