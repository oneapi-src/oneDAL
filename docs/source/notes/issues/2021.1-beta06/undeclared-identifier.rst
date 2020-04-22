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

Undeclared identifier ``CL_DEVICE_IL_VERSION_KHR``
**************************************************

When you build an application with |short_name|, you might encounter the following problem:

.. code-block:: text

    error: use of undeclared identifier CL_DEVICE_IL_VERSION_KHR
    
This is caused by a bug in |dpcpp| 2021.1-beta06 release.

How to fix
----------

The workaround for this issue is to set ``CPLUS_INCLUDE_PATH`` as follows:

.. code-block:: bash

    export CPLUS_INCLUDE_PATH="$DPCPP_ROOT/compiler/latest/linux/include/sycl"
