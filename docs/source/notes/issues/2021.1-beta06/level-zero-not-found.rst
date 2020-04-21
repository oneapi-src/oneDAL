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

.. _issue_level_zero:

Level Zero runtime dependency
*****************************

When you build an application with |short_name| on Linux\*, you might encounter the following error:

.. code-block:: text

    fatal error: 'level_zero/ze_api.h' file not found

This means that you do not have Level Zero in your environment.

.. include:: includes/note-refer-to-dpcpp-level-zero.rst

How to Fix
----------

There are two ways to fix this:

- Change your driver version to the one with Level Zero support.
- Compile your application without Level Zero support:

    1. Use ``DAAL_DISABLE_LEVEL_ZERO`` define:

       .. code-block:: cpp

         #define DAAL_DISABLE_LEVEL_ZERO

    2. Disable Level Zero runtime:
    
       .. code-block:: bash
     
         export SYCL_BE=PI_OPENCL
