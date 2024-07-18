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

oneDAL uses Intel(R) oneAPI Threading Building Blocks (Intel(R) oneTBB) to do parallel
computations on CPU.

But oneTBB is not used in the code of oneDAL algorithms directly. The algorithms rather
use custom primitives that either wrap oneTBB functionality or are inhome developed.
Those primitives form oneDAL's threading layer.

This is done in order not to be dependent on possible oneTBB API changes and even
on the particular threading technology.

The API of the layer is defined in `threading.h <https://github.com/oneapi-src/oneDAL/blob/main/cpp/daal/src/threading/threading.h>`_.

This chapter describes common parallel patterns and primitives of the threading layer.

threader_for
************

Lets consider you need to compute an elementwise sum of two arrays and store the results
into another array.
Here is a variant of sequential implementation:

.. include:: ../includes/threading/sum-sequential.rst

There are several options available in oneDAL's threaded layer to let the iterations of this code
run in parallel.
One of the options is to use `daal::threader_for` as shown here:

.. include:: ../includes/threading/sum-parallel.rst

The iteration space here goes from `0` to `n-1`.
`nThreads` is the number of threads that execute the loop's body.
And the last argument is the lambda function that defines a function object that proceeds `i`-th
iteration of the loop.
