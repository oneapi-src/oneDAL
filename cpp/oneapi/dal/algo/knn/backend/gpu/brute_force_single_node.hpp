/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#pragma once

namespace oneapi::dal::knn::backend {

using idx_t = std::int32_t;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

using voting_t = ::oneapi::dal::knn::voting_mode;

namespace de = ::oneapi::dal::detail;
namespace bk = ::oneapi::dal::backend;
namespace pr = ::oneapi::dal::backend::primitives;

template <typename Task, typename Float>
struct task_to_response_map {
    using type = int;
};

template <typename Float>
struct task_to_response_map<task::regression, Float> {
    using type = float;
};

template <typename Float>
struct task_to_response_map<task::classification, Float> {
    using type = std::int32_t;
};

template <typename Task, typename Float>
using response_t = typename task_to_response_map<Task, Float>::type;

#ifdef ONEDAL_DATA_PARALLEL

template <typename Task,
          typename Float,
          pr::ndorder torder,
          pr::ndorder qorder,
          typename RespT = response_t<Task, Float>>
sycl::event bf_kernel(sycl::queue& queue,
                      const descriptor_t<Task>& desc,
                      const pr::ndview<Float, 2, torder>& train,
                      const pr::ndview<Float, 2, qorder>& query,
                      const pr::ndview<RespT, 1>& tresps,
                      pr::ndview<Float, 2>& distances,
                      pr::ndview<idx_t, 2>& indices,
                      pr::ndview<RespT, 1>& qresps,
                      const bk::event_vector& deps = {});

#endif

} // namespace oneapi::dal::knn::backend
