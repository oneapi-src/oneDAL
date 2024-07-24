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

By default oneTBB uses dynamic work scheaduling and work stealing.
It means that

Nested parallelism
******************
