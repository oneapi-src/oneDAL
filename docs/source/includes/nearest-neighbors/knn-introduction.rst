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


:math:`k`-NN :capterm:`classification` algorithm infers the class for the new
feature vector by computing majority vote of the :math:`k` nearest observations
from the training set.


.. |t_math| replace:: :ref:`Training <knn_t_math>`
.. |t_brute_f| replace:: :ref:`Brute-force <knn_t_math_brute_force>`
.. |t_kd_tree| replace:: :ref:`k-d tree <knn_t_math_kd_tree>`
.. |t_input| replace:: :ref:`train_input <knn_t_api_input>`
.. |t_result| replace:: :ref:`train_result <knn_t_api_result>`
.. |t_op| replace:: :ref:`train(...) <knn_t_api>`

.. |i_math| replace:: :ref:`Inference <knn_i_math>`
.. |i_brute_f| replace:: :ref:`Brute-force <knn_i_math_brute_force>`
.. |i_kd_tree| replace:: :ref:`k-d tree <knn_i_math_kd_tree>`
.. |i_input| replace:: :ref:`infer_input <knn_i_api_input>`
.. |i_result| replace:: :ref:`infer_result <knn_i_api_result>`
.. |i_op| replace:: :ref:`infer(...) <knn_i_api>`

=============== ============= ============= ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|      |t_brute_f|   |t_kd_tree|   |t_op|   |t_input|   |t_result|
   |i_math|      |i_brute_f|   |i_kd_tree|   |i_op|   |i_input|   |i_result|
=============== ============= ============= ======== =========== ============
