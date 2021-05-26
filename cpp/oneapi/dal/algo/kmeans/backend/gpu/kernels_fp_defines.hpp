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

#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
#define INSTANTIATE_WITH_METRIC(F, M)                                                              \
    template sycl::event kernels_fp<F>::assign_clusters<M<F>>(sycl::queue & queue,                 \
                                                              const pr::ndview<F, 2>& data,        \
                                                              const pr::ndview<F, 2>& centroids,   \
                                                              std::int64_t block_rows,             \
                                                              pr::ndview<std::int32_t, 2>& labels, \
                                                              pr::ndview<F, 2>& distances,         \
                                                              pr::ndview<F, 2>& closest_distances, \
                                                              const bk::event_vector& deps);
#endif

} // namespace oneapi::dal::kmeans::backend
