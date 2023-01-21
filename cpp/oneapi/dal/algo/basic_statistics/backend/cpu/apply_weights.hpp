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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/util/common.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Cpu, typename Float>
std::int64_t propose_threading_block(std::int64_t row_count, std::int64_t col_count);

template <typename Cpu, typename Float>
void apply_weights_single_thread(const pr::ndview<Float, 1>& weights,
                                 pr::ndview<Float, 2>& samples);

template <typename Cpu, typename Float>
void apply_weights(const pr::ndview<Float, 1>& weights, pr::ndview<Float, 2>& samples);

template <typename Float>
void apply_weights_single_thread(const dal::backend::context_cpu& context,
                                 const pr::ndview<Float, 1>& weights,
                                 pr::ndview<Float, 2>& samples);

template <typename Float>
void apply_weights(const dal::backend::context_cpu& context,
                   const pr::ndview<Float, 1>& weights,
                   pr::ndview<Float, 2>& samples);

} // namespace oneapi::dal::basic_statistics::backend
