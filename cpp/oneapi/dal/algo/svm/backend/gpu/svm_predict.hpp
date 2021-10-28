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

#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::svm::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace pr = dal::backend::primitives;

template <typename Float>
class predict_task {
public:
    virtual ~predict_task() = default;

    virtual pr::ndarray<Float, 2> kernel_compute(sycl::queue& q,
                                                 const std::int64_t start_row,
                                                 const std::int64_t n_rows) = 0;

protected:
    predict_task(const std::int64_t max_row_per_block,
                 const table& x_table,
                 const table& sv_table,
                 const detail::kernel_function_ptr& kernel)
            : x_table_(x_table),
              sv_table_(sv_table),
              kernel_(kernel) {}

protected:
    const table& x_table_;
    const table& sv_table_;
    detail::kernel_function_ptr kernel_;
};

template <typename Float>
class predict_task_dense : public predict_task<Float> {
public:
    virtual ~predict_task_dense() = default;

    predict_task_dense(const std::int64_t max_row_per_block,
                       const table& x_table,
                       const table& sv_table,
                       const detail::kernel_function_ptr& kernel)
            : predict_task<Float>(max_row_per_block, x_table, sv_table, kernel) {}

    pr::ndarray<Float, 2> kernel_compute(sycl::queue& q,
                                         const std::int64_t start_row,
                                         const std::int64_t n_rows) {
        auto data_nd = pr::table2ndarray<Float>(q, this->x_table_);
        auto data = data_nd.get_data() + data_nd.get_dimension(1) * start_row;

        table x_block_nt = homogen_table::wrap(q, data, n_rows, data_nd.get_dimension(1));

        const auto result =
            this->kernel_->compute_kernel_function(dal::detail::data_parallel_policy(q),
                                                   x_block_nt,
                                                   this->sv_table_);

        return pr::table2ndarray<Float>(q, result, sycl::usm::alloc::device);
    }
};
}
#endif
