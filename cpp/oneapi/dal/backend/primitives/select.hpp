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

#pragma once

#include <cstdint>

#ifdef ONEAPI_DAL_DATA_PARALLEL
    #include <CL/sycl.hpp>
#endif

#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEAPI_DAL_DATA_PARALLEL

namespace impl {

template <typename Float>
struct select_small_k_l2_kernel;

} // namespace impl

template <typename Float>
struct select_small_k_l2 {
public:
    typedef std::uint32_t idx_t;
    typedef std::pair<std::int 
    typedef select_small_k_l2_kernel<Float> kernel_t;
    constexpr inline static std::int64_t elem_size = sizeof(std::uint32_t) + sizeof(Float);
    constexpr inline static int preferred_size_flag =
        std::is_same<Float, float>::value ? cl::sycl::info::device::preferred_vector_width_float
                                          : cl::sycl::info::device::preferred_vector_width_double;

public:
    select_small_k_l2(cl::sycl::queue& queue);
    cl::sycl::event operator()(const Float* cross,
                               const Float* norms,
                               const std::int64_t batch_size,
                               const std::int64_t queue_size,
                               const std::int64_t k,
                               idx_t* nearest_indices,
                               Float* nearest_distances);
    cl::sycl::event operator()(const Float* cross,
                               const Float* norms,
                               const std::int64_t batch_size,
                               const std::int64_t queue_size,
                               const std::int64_t k,
                               idx_t* nearest_indices,
                               Float* nearest_distances,
                               const std::int64_t k_width,
                               const std::int64_t yrange);
    std::pair<std::int64_t, std::int64_t> preferred_local_size(const std::int64_t k);

private:
    cl::sycl::queue& q;
    const std::int64_t max_work_group_size, max_local_size, preferred_width;
};

#endif

} // namespace oneapi::dal::backend::primitives
