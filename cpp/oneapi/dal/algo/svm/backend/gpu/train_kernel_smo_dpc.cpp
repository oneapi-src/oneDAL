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

#include "oneapi/dal/algo/svm/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::svm::backend {

template <typename Float>
struct train_kernel_gpu<Float, task::classification, method::smo> {
    train_result operator()(const dal::backend::context_gpu& ctx,
                            const descriptor_base& params,
                            const train_input& input) const {
        throw unimplemented("SVM smo method is not implemented for GPU");
        return train_result();
    }
};

template struct train_kernel_gpu<float, task::classification, method::smo>;
template struct train_kernel_gpu<double, task::classification, method::smo>;

} // namespace oneapi::dal::svm::backend
