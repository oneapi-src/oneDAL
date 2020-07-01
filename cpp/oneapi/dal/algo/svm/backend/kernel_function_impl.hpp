/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/svm/common.hpp"

#include <daal/include/algorithms/kernel_function/kernel_function_linear.h>
#include <daal/include/algorithms/kernel_function/kernel_function_rbf.h>

namespace oneapi::dal::svm::detail {

class kernel_function_impl : public base {
public:
    virtual ~kernel_function_impl() = default;

    virtual daal::algorithms::kernel_function::KernelIfacePtr get_interop_kernel() = 0;
};

} // namespace oneapi::dal::svm::detail
