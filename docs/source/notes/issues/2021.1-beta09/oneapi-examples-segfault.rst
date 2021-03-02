.. ******************************************************************************
.. * Copyright 2020-2021 Intel Corporation
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

oneAPI examples failing with segfault
*************************************

oneAPI examples located in the ``examples/oneapi`` directory are failing with a segmentation fault (segfault).
This happens because the directory that contains data is skipped during installation.

How to Fix
----------

The workaround is to copy the missing directory from the source repository:

#. Copy the `missing data directory <https://github.com/oneapi-src/oneDAL/tree/rls/onedal-beta09-rls/examples/oneapi/data>`_ from |short_name| repository.

#. Place the ``data`` directory within the ``examples/oneapi`` directory on your machine.
