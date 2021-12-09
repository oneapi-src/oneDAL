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

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::pca::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float>
class local_result {
    using alloc = sycl::usm::alloc;
    using own_t = local_result<Float>;

public:
    static own_t empty(sycl::queue& q, std::int64_t count) {
        own_t res;
        res.rsum_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        return res;
    }

    auto& get_sum() const {
        return rsum_;
    }

private:
    local_result() = default;
    pr::ndarray<Float, 1> rsum_;
};

template <typename Float>
class local_buffer_list {
    using alloc = sycl::usm::alloc;
    using own_t = local_buffer_list<Float>;

public:
    static own_t empty(sycl::queue& q, std::int64_t count) {
        own_t res;
        res.rrow_count_ = pr::ndarray<std::int64_t, 1>::empty(q, { count }, alloc::device);
        res.rsum_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        return res;
    }
    auto& get_rc_list() const {
        return rrow_count_;
    }
    auto& get_sum() const {
        return rsum_;
    }

private:
    local_buffer_list() = default;

    pr::ndarray<std::int64_t, 1> rrow_count_;
    pr::ndarray<Float, 1> rsum_;
};

template <typename Float>
class train_kernel_cov_impl {
    using method_t = method::cov;
    using task_t = task::dim_reduction;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using input_t = train_input<task::dim_reduction>;
    using result_t = train_result<task::dim_reduction>;
    using local_result_t = local_result<Float>;
    using local_buffer_list_t = local_buffer_list<Float>;
    using descriptor_t = detail::descriptor_base<task::dim_reduction>;

public:
    train_kernel_cov_impl(const bk::context_gpu& ctx)
            : q_(ctx.get_queue()),
              comm_(ctx.get_communicator()) {}
    result_t operator()(const descriptor_t& desc, const input_t& input);

private:
    std::int64_t get_row_block_count(std::int64_t row_count);

    std::int64_t get_column_block_count(std::int64_t column_count);

    std::tuple<local_result_t, sycl::event> compute_single_pass(const pr::ndarray<Float, 2> data);

    std::tuple<local_result_t, sycl::event> compute_by_blocks(const pr::ndarray<Float, 2> data,
                                                              std::int64_t row_block_count);
    std::tuple<local_result_t, sycl::event> merge_blocks(local_buffer_list_t&& ndbuf,
                                                         std::int64_t column_count,
                                                         std::int64_t block_count,
                                                         const bk::event_vector& deps = {});
    std::tuple<local_result_t, sycl::event> merge_distr_blocks(
        const pr::ndarray<std::int64_t, 1>& com_row_count,
        const pr::ndarray<Float, 1>& com_sum,
        local_result_t&& ndres,
        std::int64_t block_count,
        std::int64_t column_count,
        std::int64_t block_stride,
        const bk::event_vector& deps = {});

    std::tuple<local_result_t, sycl::event> finalize(local_result_t&& ndres,
                                                     std::int64_t row_count,
                                                     std::int64_t column_count,
                                                     const bk::event_vector& deps = {});

    result_t get_result(const descriptor_t& desc,
                        const local_result_t& ndres,
                        std::int64_t column_count,
                        const bk::event_vector& deps = {});

    sycl::queue q_;
    comm_t comm_;
};

} // namespace oneapi::dal::pca::backend
#endif // ONEDAL_DATA_PARALLEL
