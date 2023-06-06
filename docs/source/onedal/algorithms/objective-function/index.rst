.. ******************************************************************************
.. * Copyright 2023 Intel Corporation
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

==========
Objective function
==========

Some classification algorithms are designed to minimize the 
selected objective function. On each iteration its' gradient and sometimes
hessian is calculated and model weights are updated using this information.

.. toctree::
   :titlesonly:

   logloss.rst

.. rubric:: Examples: Logistic Loss

.. include::  ../../../includes/objective-function/logloss-examples.rst
