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

#include "oneapi/dal/algo/basic_statistics/backend/cpu/apply_weights.hpp"

namespace oneapi::dal::basic_statistics::backend {

template<typename Float> 
void apply_weights_single_thread(const dal::backend::context_cpu& context,
                                 const pr::ndview<Float, 1>& weights,
                                 pr::ndview<Float, 2>& samples) {
    return dal::backend::dispatch_by_cpu(context, [&](auto cpu) {
        return apply_weights_single_thread<decltype(cpu), Float>(weights, samples);
    });
}

} // namespace oneapi::dal::basic_statistics::backend
