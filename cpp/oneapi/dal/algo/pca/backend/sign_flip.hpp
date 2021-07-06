/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::pca::backend {

template <typename Cpu, typename Float>
void sign_flip_impl(Float* eigvecs, std::int64_t row_count, std::int64_t column_count);

template <typename Float>
void sign_flip(dal::backend::primitives::ndview<Float, 2>& eigvecs) {
    using dal::backend::context_cpu;
    using dal::backend::dispatch_by_cpu;

    dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        sign_flip_impl<decltype(cpu)>(eigvecs.get_mutable_data(),
                                      eigvecs.get_dimension(0),
                                      eigvecs.get_dimension(1));
    });
}

} // namespace oneapi::dal::pca::backend
