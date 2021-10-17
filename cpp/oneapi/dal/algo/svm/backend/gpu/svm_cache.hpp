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

namespace pr = dal::backend::primitives;

template <typename Float>
class sub_data_task_base {
public:
    virtual ~sub_data_task_base() = default;
    virtual sycl::event copy_data_by_indices(sycl::queue& q,
                                             const pr::ndview<std::uint32_t, 1>& ws_indices,
                                             const std::int64_t subset_vectors_count,
                                             const pr::ndview<Float, 2>& x) = 0;
    table get_table() const {
        return data_table_;
    }

protected:
    sub_data_task_base(sycl::queue& q,
                       const std::int64_t row_count,
                       const std::int64_t column_count) {
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
                        const std::int64_t max_subset_vectors_count,
                        const std::int64_t features_count)
            : sub_data_task_base<Float>(q, max_subset_vectors_count, features_count) {
        this->data_table_ = homogen_table::wrap(this->data_nd_.flatten(q),
                                                max_subset_vectors_count,
                                                features_count);
    }

    sycl::event copy_data_by_indices(sycl::queue& q,
                                     const pr::ndview<std::uint32_t, 1>& ws_indices,
                                     const std::int64_t subset_vectors_count,
                                     const pr::ndview<Float, 2>& x) override {
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

    virtual pr::ndarray<Float, 2> compute(const dal::backend::context_gpu& ctx,
                                          const detail::kernel_function_ptr& kernel_ptr,
                                          const table& x_table,
                                          const pr::ndarray<Float, 2>& x_nd,
                                          const pr::ndview<std::uint32_t, 1>& ws_indices) = 0;

protected:
    svm_cache_iface(sycl::queue& q, const std::int64_t block_size, const std::int64_t line_size)
            : q_(q),
              block_size_(block_size),
              line_size_(line_size) {}

    sycl::queue q_;

    const std::int64_t block_size_;
    const std::int64_t line_size_;
};

template <svm_cache_type CacheType, typename Float>
class svm_cache {};

template <typename Float>
class svm_cache<no_cache, Float> : public svm_cache_iface<Float> {
public:
    svm_cache(sycl::queue& q,
              const pr::ndarray<Float, 2>& data_nd,
              const double cache_size,
              const std::int64_t block_size,
              const std::int64_t line_size)
            : svm_cache_iface<Float>(q, block_size, line_size) {
        sub_data_task_ptr_ =
            std::make_shared<sub_data_task_dense<Float>>(q, block_size, data_nd.get_dimension(1));
    }

    pr::ndarray<Float, 2> compute(const dal::backend::context_gpu& ctx,
                                  const detail::kernel_function_ptr& kernel_ptr,
                                  const table& x_table,
                                  const pr::ndarray<Float, 2>& x_nd,
                                  const pr::ndview<std::uint32_t, 1>& ws_indices) override {
        const std::int64_t work_elements_count = ws_indices.get_count();
        auto copy_event = sub_data_task_ptr_->copy_data_by_indices(this->q_,
                                                                   ws_indices,
                                                                   work_elements_count,
                                                                   x_nd);

        const auto result =
            kernel_ptr->compute_kernel_function(ctx, sub_data_task_ptr_->get_table(), x_table);

        return pr::table2ndarray<Float>(this->q_, result, sycl::usm::alloc::device);
    }

private:
    std::shared_ptr<sub_data_task_base<Float>> sub_data_task_ptr_;
};

} // namespace oneapi::dal::svm::backend
