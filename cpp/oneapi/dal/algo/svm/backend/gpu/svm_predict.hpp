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
//#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"
//#include "oneapi/dal/algo/svm/backend/gpu/svm_predict.hpp"

namespace oneapi::dal::svm::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace pr = dal::backend::primitives;

template <typename Float>
class predict_task {
public:
    virtual ~predict_task() = default;

    pr::ndarray<Float, 2> kernel_compute(sycl::queue& q, const std::int64_t start_row, const std::int64_t n_rows)
    {
        auto x_block_nt = get_block_nt_data(start_row, n_rows);

        // std::cout << x_block_nt.get_row_count() << " ";
        // std::cout << x_block_nt.get_column_count() << "\n";

        const auto result =
            kernel_->compute_kernel_function(dal::detail::data_parallel_policy(q),
                                                x_block_nt,
                                                sv_table_);

        return pr::table2ndarray<Float>(q, result, sycl::usm::alloc::device);
    }

protected:
    predict_task(sycl::queue& q, const std::int64_t max_row_per_block, const table & x_table, const table & sv_table,
                const detail::kernel_function_ptr & kernel)
        : x_table_(x_table), sv_table_(sv_table), _n_size(sv_table.get_row_count()), queue_(q)
    {
        kernel_ = kernel;
    }

    virtual table get_block_nt_data(const std::int64_t start_row, const std::int64_t n_rows) = 0;

protected:
    const table& x_table_;
    const table& sv_table_;
    std::int64_t _n_size;
    detail::kernel_function_ptr kernel_;
    sycl::queue queue_;
};

template <typename Float>
class predict_task_dense : public predict_task<Float>
{
public:
    virtual ~predict_task_dense() = default;

    predict_task_dense(sycl::queue& q, const std::int64_t max_row_per_block, const table& x_table, const table & sv_table,
                     const detail::kernel_function_ptr & kernel)
        : predict_task<Float>(q, max_row_per_block, x_table, sv_table, kernel)
    {}

    table get_block_nt_data(const std::int64_t start_row, const std::int64_t n_rows) override
    {
        auto data_nd = pr::table2ndarray<Float>(this->queue_, this->x_table_);
        auto data = data_nd.get_data() + data_nd.get_dimension(1) * start_row;


        table x_block_nt = homogen_table::wrap(data, n_rows, data_nd.get_dimension(1));
        return x_block_nt;
    }

};
}
#endif
