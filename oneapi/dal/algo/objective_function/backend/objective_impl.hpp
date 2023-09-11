/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

// #include <daal/src/algorithms/service_kernel_math.h>

namespace oneapi::dal::objective_function::detail {

class objective_impl : public base {
public:
    virtual ~objective_impl() = default;
    virtual double get_l1_regularization_coefficient() = 0;
    virtual double get_l2_regularization_coefficient() = 0;
    virtual bool get_intercept_flag() = 0;
};

} // namespace oneapi::dal::objective_function::detail
