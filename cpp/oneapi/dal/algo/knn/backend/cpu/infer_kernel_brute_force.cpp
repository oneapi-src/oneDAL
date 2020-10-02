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

#include "oneapi/dal/algo/knn/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/detail/common.hpp"

#define DAAL_SYCL_INTERFACE

namespace oneapi::dal::knn::backend {

using dal::backend::context_cpu;
template <typename Float>
struct infer_kernel_cpu<Float, method::brute_force> {
    infer_result operator()(const context_cpu &ctx,
                            const descriptor_base &desc,
                            const infer_input &input) const {
        throw unimplemented("k-NN brute force method is not implemented for CPU");
        return infer_result();
    }
};

template struct infer_kernel_cpu<float, method::brute_force>;
template struct infer_kernel_cpu<double, method::brute_force>;

} // namespace oneapi::dal::knn::backend
