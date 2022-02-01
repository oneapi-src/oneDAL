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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

sycl::event count_clusters(sycl::queue& queue,
                           const pr::ndview<std::int32_t, 2>& responses,
                           std::int64_t cluster_count,
                           pr::ndview<std::int32_t, 1>& counters,
                           const bk::event_vector& deps = {});

std::int64_t count_empty_clusters(sycl::queue& queue,
                                  std::int64_t cluster_count,
                                  pr::ndview<std::int32_t, 1>& counters,
                                  const bk::event_vector& deps = {});
#endif

} // namespace oneapi::dal::kmeans::backend
