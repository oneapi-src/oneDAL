.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
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

.. default-domain:: cpp

.. _api_linear-regression:

=================
Linear Regression
=================

.. include::  ../../../includes/linear-regression/linear-regression-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Linear Regression <alg_linear-regression>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::linear_regression`` namespace and are available via inclusion of the
``oneapi/dal/algo/linear_regression.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::linear_regression::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::linear_regression::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::linear_regression::task

.. _linear-regression_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _linear-regression_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::linear_regression::compute_input

.. _linear-regression_c_api_result:

Result and Finalize Result
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::linear_regression::compute_result


.. _linear-regression_p_api:

Partial Training :expr:`partial_compute(...)`
---------------------------------------------
.. _linear-regression_p_api_input:

Partial Input
~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::linear_regression::partial_compute_input

.. _linear-regression_p_api_result:

Partial Result and Finalize Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::linear_regression::partial_compute_result

Finalize Training :expr:`finalize_compute(...)`
-----------------------------------------------
