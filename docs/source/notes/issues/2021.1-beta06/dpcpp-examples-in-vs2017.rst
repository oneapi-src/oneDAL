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

DPC++ examples not working
**************************

DPC++ examples for |short_name| might not work in Visual Studio 2017 by default. You will see the following message:

.. code-block:: text

    The Windows SDK version 10.0 was not found. Install the required version of Windows SDK or change the SDK version 
    in the project property pages or by right-clicking the solution and selecting "Retarget solution".
 
How to Fix
----------

Choose the installed SDK version and select :guilabel:`Retarget solution`.
