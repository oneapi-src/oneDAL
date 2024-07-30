.. ******************************************************************************
.. * Copyright contributors to the oneDAL project
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. highlight:: cpp

Threading Layer
^^^^^^^^^^^^^^^

oneDAL uses Intel\ |reg|\  oneAPI Threading Building Blocks (Intel\ |reg|\  oneTBB) to do parallel
computations on CPU.

But oneTBB is not used in the code of oneDAL algorithms directly. The algorithms rather
use custom primitives that either wrap oneTBB functionality or are inhome developed.
Those primitives form oneDAL's threading layer.

This is done in order not to be dependent on possible oneTBB API changes and even
on the particular threading technology.

The API of the layer is defined in
`threading.h <https://github.com/oneapi-src/oneDAL/blob/main/cpp/daal/src/threading/threading.h>`_.
Please be aware that those APIs are not publicly defined. So they can be changed at any time
without any notification.

This chapter describes common parallel patterns and primitives of the threading layer.

threader_for
************

Lets consider you need to compute an elementwise sum of two arrays and store the results
into another array.
Here is a variant of sequential implementation:

.. include:: ../includes/threading/sum-sequential.rst

There are several options available in oneDAL's threading layer to let the iterations of this code
run in parallel.
One of the options is to use ``daal::threader_for`` as shown here:

.. include:: ../includes/threading/sum-parallel.rst

The iteration space here goes from ``0`` to ``n-1``.
The last argument is the lambda function that defines a function object that proceeds ``i``-th
iteration of the loop.

Blocking
--------

To have more control over the parallel execution and to increase
`cache locality <https://en.wikipedia.org/wiki/Locality_of_reference>`_ oneDAL usually splits
the data into blocks and then processes those blocks in parallel.

This code shows how a typical parallel loop in oneDAL looks like:

.. include:: ../includes/threading/sum-parallel-by-blocks.rst

Thread-local Storage (TLS)
**************************

Lets consider you need to compute a dot product of two arrays.
Here is a variant of sequential implementation:

.. include:: ../includes/threading/dot-sequential.rst

Parallel computations can be performed in two steps:

    1. Compute partial dot product at each threaded.
    2. Perform a reduction: Sum the partial results from all threads to compute the final dot product.

``daal::tls`` provides a local storage where each thread can accumulate its local results.
Following code allocates memory that would store partial dot products for each thread:

.. include:: ../includes/threading/dot-parallel-init-tls.rst

``SafeStatus`` in this code denotes a thread-safe counterpart of oneDAL's ``Status`` class.
``SafeStatus`` allows to collect errors from all threads and report them to user using
``detach()`` method as it will be shown later in the code.

Checking the status right after the initialization code won't show the allocation errors though.
Because oneTBB uses lazy evaluation and the lambda function passed to the constructor of the TLS
is evaluated in the moment of the TLS's first use.

Again, there are several options available in oneDAL's threading layer to compute the partial
dot product results at each thread.
One of the options is to use already mentioned ``daal::threader_for`` and blocking approach
as shown here:

.. include:: ../includes/threading/dot-parallel-partial-compute.rst

To compute the final result it is requred to reduce TLS's partial results over all threads
as it is shown here:

.. include:: ../includes/threading/dot-parallel-reduction.rst

Local memory of the threads should also be released when it is no longer needed.

The complete parallel verision of dot product computations would look like:

.. include:: ../includes/threading/dot-parallel.rst

Static work scheduling
**********************

By default oneTBB uses
`dynamic work scheduling <https://oneapi-src.github.io/oneTBB/main/tbb_userguide/How_Task_Scheduler_Works.html>`_
and work stealing.
It means that two different runs of the same parallel loop can produce different
mappings of the loop's iteration space to the available threads.
This strategy is benefitial when it is hard to estimate the amount of work performed
by each iteration.

In the cases when it is known that the iterations perform equal amount of work it
might be benefitial to use pre-defined mapping of the loop's iterations to threads.
This is what static work scheduling does.

``daal::static_threader_for`` and ``daal::static_tls`` allow to implement static
work scheduling within oneDAL.

Here is a variant of parallel dot product computation with static scheduling:

.. include:: ../includes/threading/dot-static-parallel.rst

Nested parallelism
******************

It is allowed to have nested parallel loops within oneDAL.
What is important to know is that

    "when a parallel construct calls another parallel construct, a thread can obtain a task
     from the outer-level construct while waiting for completion of the inner-level one."

    -- `oneTBB documentation <https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2021-13/work-isolation.html>`_

In practice this means that, for example, a thread-local variable might unexpectedly
change its value after a nested parallel construct:

.. include:: ../includes/threading/nested-parallel.rst

In some scenarios this can lead to deadlocks, segmentation faults and other issues.

oneTBB provides ways to isolate execution of a parallel construct, for its tasks
to not interfere with other simultaneously running tasks.

Those options are preferred when the parallel loops are initially written as nested.
But in oneDAL there are cases when one parallel algorithm, the outer one,
calls another parallel algorithm, the inner one, within a parallel region.

The inner algorithm in this case can also be called solely, without additional nesting.
And we do not want to always make it isolated.

For the cases like that oneDAL provides ``daal::ls``. Its ``local()`` method always
returns the same value for the same thread, regardless of the nested execution:

.. include:: ../includes/threading/nested-parallel-ls.rst
