.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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

Potential performance degradations
**********************************

You might experience performance degradations if you use Lever Zero Runtime for GPU computations.

.. include:: includes/note-refer-to-dpcpp-level-zero.rst

How to Fix
----------

Switch back to OpenCL runtime by setting the ``SYCL_BE`` variable to ``PI_OPENCL``:

.. code-block:: bash

    export SYCL_BE=PI_OPENCL
