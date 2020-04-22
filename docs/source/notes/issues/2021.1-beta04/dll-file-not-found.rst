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

DLL file not found
******************

If you run your program in Visual Studio and encounter a "sycl.dll was not found" runtime error
or a similar one such as the one shown below, update the project property :guilabel:`Debugging` > :guilabel:`Environment`. 
To do this, follow `How to Fix`_ instructions.

  .. image:: images/runtime_error.png
    :alt: Unable to start a program: the code execution cannot proceed because XXX.dll was not found. 
    :class: with-border

.. attention::

  The issue appears as the result of how Visual Studio 2017 and its later versions
  handle additional directories for the ``PATH`` environment variable.

How to Fix
----------

1. Open the project's properties, go to :guilabel:`Debugging` > :guilabel:`Environment` property, 
   right-click the drop-down menu, and select :guilabel:`Edit`:

  .. image:: images/vsproj_debug_step1_open.png
    :width: 600
    :alt: Changing configuration properties
    :class: with-border

2. Copy the default value of the ``PATH`` environment variable from the :guilabel:`Evaluated value`
   section and then paste it to the section above it:

  .. image:: images/vsproj_debug_step2_copy.png
    :width: 600
    :alt: Changing configuration properties
    :class: with-border

3. Add to ``PATH`` the paths to the dll files that the program needs:

  .. image:: images/vsproj_debug_step3_add.png
    :width: 600
    :alt: Changing configuration properties
    :class: with-border
