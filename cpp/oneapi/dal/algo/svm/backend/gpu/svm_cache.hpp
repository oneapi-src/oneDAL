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
#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"

namespace oneapi::dal::svm::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace pr = dal::backend::primitives;

template <typename Float>
class sub_data_task_base {
public:
    virtual ~sub_data_task_base() = default;
    virtual sycl::event copy_data_by_indices(sycl::queue& q,
                                             const pr::ndview<std::int32_t, 1>& ws_indices,
                                             const std::int32_t subset_vectors_count,
                                             const pr::ndview<Float, 2>& x) = 0;
    table get_table() const {
        return data_table_;
    }

protected:
    sub_data_task_base(sycl::queue& q,
                       const std::int32_t row_count,
                       const std::int32_t column_count) {
        data_nd_ =
            pr::ndarray<Float, 2>::empty(q, { row_count, column_count }, sycl::usm::alloc::device);
    }

    sub_data_task_base() {}

    table data_table_;
    pr::ndarray<Float, 2> data_nd_;
};

template <typename Float>
class sub_data_task_dense : public sub_data_task_base<Float> {
public:
    sub_data_task_dense(sycl::queue& q,
                        const std::int32_t max_subset_vectors_count,
                        const std::int32_t features_count)
            : sub_data_task_base<Float>(q, max_subset_vectors_count, features_count) {
        this->data_table_ = homogen_table::wrap(this->data_nd_.flatten(q),
                                                max_subset_vectors_count,
                                                features_count);
    }

    sycl::event copy_data_by_indices(sycl::queue& q,
                                     const pr::ndview<std::int32_t, 1>& ws_indices,
                                     const std::int32_t subset_vectors_count,
                                     const pr::ndview<Float, 2>& x) override {
        ONEDAL_PROFILER_TASK(cache_compute.copy_data_by_indices, q);
        auto event = copy_by_indices(q,
                                     x,
                                     ws_indices,
                                     this->data_nd_,
                                     subset_vectors_count,
                                     x.get_dimension(1));
        return event;
    }
};

enum svm_cache_type {
    no_cache, /*!< No storage for caching kernel function values is provided */
    simple_cache, /*!< Storage for caching ALL kernel function values is provided */
    lru_cache /*!< Storage for caching PART  of kernel function values is provided;
                         LRU algorithm is used to exclude values from cache */
};

template <typename Float>
class svm_cache_iface {
public:
    virtual ~svm_cache_iface() = default;

    virtual pr::ndarray<Float, 2> compute(const detail::kernel_function_ptr& kernel_ptr,
                                          const table& x_table,
                                          const pr::ndarray<Float, 2>& x_nd,
                                          const pr::ndview<std::int32_t, 1>& ws_indices) = 0;

protected:
    svm_cache_iface() {}
};

template <svm_cache_type CacheType, typename Float>
class svm_cache {};

template <typename Float>
class svm_cache<no_cache, Float> : public svm_cache_iface<Float> {
public:
    svm_cache(sycl::queue& q,
              const pr::ndarray<Float, 2>& data_nd,
              const double cache_size,
              const std::int32_t block_size,
              const std::int32_t line_size)
            : svm_cache_iface<Float>(),
              q_(q) {
        sub_data_task_ptr_ =
            std::make_shared<sub_data_task_dense<Float>>(q, block_size, data_nd.get_dimension(1));
        res_nd_ =
            pr::ndarray<Float, 2>::empty(q, { block_size, line_size }, sycl::usm::alloc::device);
        res_table_ = homogen_table::wrap(q, res_nd_.get_mutable_data(), block_size, line_size);
    }

    pr::ndarray<Float, 2> compute(const detail::kernel_function_ptr& kernel_ptr,
                                  const table& x_table,
                                  const pr::ndarray<Float, 2>& x_nd,
                                  const pr::ndview<std::int32_t, 1>& ws_indices) override {
        ONEDAL_PROFILER_TASK(cache_compute, this->q_);
        const std::int32_t work_elements_count = ws_indices.get_count();
        sub_data_task_ptr_->copy_data_by_indices(this->q_, ws_indices, work_elements_count, x_nd)
            .wait_and_throw();

        kernel_ptr->compute_kernel_function(dal::detail::data_parallel_policy(this->q_),
                                            sub_data_task_ptr_->get_table(),
                                            x_table,
                                            res_table_);

        return res_nd_;
    }

private:
    sycl::queue q_;
    std::shared_ptr<sub_data_task_base<Float>> sub_data_task_ptr_;
    homogen_table res_table_;
    pr::ndarray<Float, 2> res_nd_;
};

#endif

} // namespace oneapi::dal::svm::backend
