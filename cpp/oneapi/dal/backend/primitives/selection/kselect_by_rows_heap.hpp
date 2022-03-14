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
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_base.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_data_provider.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
std::int64_t get_heap_min_k(const sycl::queue& q);

template <typename Float>
std::int64_t get_heap_max_k(const sycl::queue& q);

std::int64_t get_preferred_sub_group(const sycl::queue& q);

template <typename Float, bool dst_out, bool ids_out, int sg_size = 64>
sycl::event select(sycl::queue& queue,
                   const ndview<Float, 2>& data,
                   std::int64_t k,
                   ndview<Float, 2>& selection,
                   ndview<std::int32_t, 2>& indices,
                   const event_vector& deps);

template <typename Float, bool dst_out, bool ids_out, int sg_size = 64>
sycl::event sq_l2_select(sycl::queue& queue,
                         const ndview<Float, 1>& n1,
                         const ndview<Float, 1>& n2,
                         const ndview<Float, 2>& ip,
                         std::int64_t k,
                         ndview<Float, 2>& selection,
                         ndview<std::int32_t, 2>& indices,
                         const event_vector& deps);

// Performs k-selection for medium k-values
template <typename Float>
class kselect_by_rows_heap : public kselect_by_rows_base<Float> {
    using sq_l2_dp_t = data_provider_t<Float, true>;
    using naive_dp_t = data_provider_t<Float, false>;

public:
    kselect_by_rows_heap();
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           ndview<std::int32_t, 2>& indices,
                           const event_vector& deps) override;

    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           const event_vector& deps) override;

    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<std::int32_t, 2>& indices,
                           const event_vector& deps) override;

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<Float, 2>& selection,
                             ndview<std::int32_t, 2>& indices,
                             const event_vector& deps) override;

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<Float, 2>& selection,
                             const event_vector& deps) override;

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<std::int32_t, 2>& indices,
                             const event_vector& deps) override;
};
#endif

} // namespace oneapi::dal::backend::primitives
