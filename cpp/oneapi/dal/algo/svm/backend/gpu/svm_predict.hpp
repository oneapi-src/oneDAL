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

    virtual pr::ndarray<Float, 2> kernel_compute(const std::int64_t start_row,
                                                 const std::int64_t n_rows,
                                                 const std::int64_t sv_count) = 0;

protected:
    predict_task(sycl::queue& q,
                 const std::int64_t max_row_per_block,
                 const pr::ndarray<Float, 2>& data_nd,
                 const table& sv_table,
                 const detail::kernel_function_ptr& kernel)
            : q_(q),
              data_nd_(data_nd),
              sv_table_(sv_table),
              kernel_(kernel) {
        res_nd_ = pr::ndarray<Float, 1>::empty(q,
                                               { max_row_per_block * sv_table_.get_row_count() },
                                               sycl::usm::alloc::device);
    }

protected:
    sycl::queue q_;
    const pr::ndarray<Float, 2> data_nd_;
    const table sv_table_;
    homogen_table res_table_;
    pr::ndarray<Float, 1> res_nd_;
    detail::kernel_function_ptr kernel_;
};

template <typename Float>
class predict_task_dense : public predict_task<Float> {
public:
    virtual ~predict_task_dense() = default;

    predict_task_dense(sycl::queue& q,
                       const std::int64_t max_row_per_block,
                       const pr::ndarray<Float, 2>& data_nd,
                       const table& sv_table,
                       const detail::kernel_function_ptr& kernel)
            : predict_task<Float>(q, max_row_per_block, data_nd, sv_table, kernel) {}

    pr::ndarray<Float, 2> kernel_compute(const std::int64_t start_row,
                                         const std::int64_t n_rows,
                                         const std::int64_t sv_count) {
        auto data = this->data_nd_.get_data() + this->data_nd_.get_dimension(1) * start_row;

        table x_block_nt =
            homogen_table::wrap(this->q_, data, n_rows, this->data_nd_.get_dimension(1));

        this->res_table_ =
            homogen_table::wrap(this->q_, this->res_nd_.get_mutable_data(), n_rows, sv_count);
        this->kernel_->compute_kernel_function(dal::detail::data_parallel_policy(this->q_),
                                               x_block_nt,
                                               this->sv_table_,
                                               this->res_table_);

        return pr::ndarray<Float, 2>::wrap(this->res_nd_.get_data(), { n_rows, sv_count });
    }
};
}
#endif
