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

#include "oneapi/dal/algo/rbf_kernel/backend/gpu/compute_kernel.hpp"

namespace oneapi::dal::rbf_kernel::backend {

template <typename Float>
struct compute_kernel_gpu<Float, method::csr> {
    compute_result operator()(const dal::backend::context_gpu& ctx,
                              const descriptor_base& params,
                              const compute_input& input) const {
        return compute_result();
    }
};

template struct compute_kernel_gpu<float, method::csr>;
template struct compute_kernel_gpu<double, method::csr>;

} // namespace oneapi::dal::rbf_kernel::backend
