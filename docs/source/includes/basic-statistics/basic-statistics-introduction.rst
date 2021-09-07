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

Basic statistics algorithm computes the following set of quantitative dataset characteristics: 

- minimums/maximums
- sums
- means
- sums of squares
- sums of squared differences from the means
- second order raw moments
- variances
- standard deviations
- variations

.. |c_math| replace::   `dense <basic_statistics_c_math_>`_
.. |c_dense| replace::  `dense <basic_statistics_c_math_dense_>`_
.. |c_input| replace::  `compute_input <basic_statistics_c_api_input_>`_
.. |c_result| replace:: `compute_result <basic_statistics_c_api_result_>`_
.. |c_op| replace::     `compute(...) <basic_statistics_c_api_>`_

=============  ===============  =========  =============  ===========
**Operation**  **Computational  methods**  **Programming  Interface**
-------------  --------------------------  --------------------------
|c_math|       |c_dense|        |c_op|     |c_input|      |c_result|
=============  ===============  =========  =============  ===========
