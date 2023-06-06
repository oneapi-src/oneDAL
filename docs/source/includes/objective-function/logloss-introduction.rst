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




LogisticLoss is a common objective function used for binary classification. 
LogisticLoss algorithm provides the following functionality.
- Value
- Gradient
- Hessian

.. |c_math| replace::   :ref:`dense_batch <logloss_c_math>`
.. |c_dense| replace::  :ref:`dense_batch <logloss_c_dense_batch>`
.. |c_input| replace::  :ref:`compute_input <logloss_c_api_input>`
.. |c_result| replace:: :ref:`compute_result <logloss_c_api_result>`
.. |c_op| replace::     :ref:`compute(...) <logloss_c_api>`

=============  ===============  =========  =============  ===========
**Operation**  **Computational  methods**  **Programming  Interface**
-------------  --------------------------  --------------------------
|c_math|       |c_dense|        |c_op|     |c_input|      |c_result|
=============  ===============  =========  =============  ===========
