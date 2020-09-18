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

PCA examples failing on GPU devices
***********************************

Prinical Component Analysis (PCA) examples are failing on GPU devices but work on CPU-only systems.

.. code-block:: text

    terminate called after throwing an instance of 'oneapi::dal::internal_error'
    what():  Result eigenvalues should not be empty

How to Fix
----------

There is no workaround available.

You can evaluate examples located in the ``examples/daal/cpp_sycl`` directory.
They provide oneAPI support and work on both CPU and GPU devices.
