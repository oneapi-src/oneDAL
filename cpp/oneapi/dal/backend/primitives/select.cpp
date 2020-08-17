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

#include <algorithm>
#include <utility>
#include <exception>
#include <limits>
#include <list>

#ifdef ONEAPI_DAL_DATA_PARALLEL
    #include <CL/sycl.hpp>
#endif

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/select.hpp"
#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEAPI_DAL_DATA_PARALLEL

namespace impl {

template <typename Float>
struct select_small_k_l2_kernel {
public:
    typedef std::uint32_t idx_t;
    typedef cl::sycl::
        accessor<Float, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
            dist_acc_t;
    typedef cl::sycl::
        accessor<idx_t, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
            idxs_acc_t;

public:
    select_small_l2_kernel(const Float* input_cross_,
                           const Float* input_norms_,
                           idx_t* output_indices_,
                           Float* output_distances_,
                           idx_t k_,
                           idx_t n_vectors_,
                           idx_t k_width_,
                           idx_t vector_size_;
                           dist_acc_t dwspace_, idxs_acc_t iwspace_)
            : input_cross(input_cross_),
              input_norms(input_norms_),
              output_indices(output_indices_),
              output_distances(output_distances_),
              k(k_),
              n_vectors(n_vectors_),
              k_width(k_with_),
              vector_size(vector_size_),
              dwspace(dwspace_),
              iwspace(iwspace_){};

    void operator()(cl::sycl::nd_item<2> idx) {
        const idx_t lxid = idx.get_local_id(0);
        const idx_t lyid = idx.get_local_id(1);

        const idx_t xrange = idx.get_local_range(0);
        const idx_t yrange = idx.get_local_range(1);
        // Working row
        const idx_t gyid = idx.get_global_id(1);

        // Shift in local workspace
        const idx_t lrid = lyid * k_width;

        // Copying from output to loacl buffer to enable merging
        {
            if (gyid < n_vectors) {
                const idx_t* const irow = &(output_indices[gyid * k]);
                const Float* const drow = &(ouput_distances[gyid * k]);
                for (idx_t shift = lxid; shift < k; shift += xrange) {
                    iwspace[lrid + shift] = irow[shift];
                    dwspace[lrid + shift] = drow[shift];
                }
            }
            idx.barrier(cl::sycl::access::fence_space::local_space);
            // TODO: Optimize with async_work_group_copy
        }

        {
            const idx_t k_step = k_width - k;
            // Loop along vector size
            for (idx_t shift = lxid; shift < vector_size; shift += k_step) {
                // Copying data
                for (idx_t i = 0; i < k_step; i += xrange) {
                    iwspace[lrid + k + i] = shift + i;
                    dwspace[lrid + k + i] =
                        input_norms[shift + i] - 2 * input_cross[vector_size * gyid + shift + i];
                }
                idx.barrier(cl::sycl::access::fence_space::local_space);
                // TODO: Another sort?
                // Odd-Even Sorting
                {
                    // Sorting itself
                    for (idx_t step = 0; step < k_width; ++step) {
                        for (idx_t i = 2 * lxid + step & 1; i + 1 < k_width; i += xrange) {
                            const bool swap = dwspace[lrid + i] > dwspace[lrid + i + 1];
                            conditional_swap<Float>(dwspace[lrid + i], dwspace[lrid + i + 1], swap);
                            conditional_swap<idx_t>(iwspace[lrid + i], iwspace[lrid + i + 1], swap);
                        }
                        idx.barrier(cl::sycl::access::fence_space::local_space);
                    }
                }
            }
        }

        //Exit from algorithm
        {
            if (gyid < n_vectors) {
                const idx_t* const irow = &(output_indices[gyid * k]);
                const Float* const drow = &(ouput_distances[gyid * k]);
                for (idx_t shift = lxid; shift < k; shift += xrange) {
                    irow[shift] = iwspace[lrid + shift];
                    drow[shift] = dwspace[lrid + shift];
                }
            }
            // TODO: Optimize with async_work_group_copy
        }
    }

private:
    const Float* const input_cross;
    const Float* const input_norms;
    idx_t* const output_indices;
    Float* const output_distances;
    const idx_t k, n_vectors, k_width, vector_size;
    dist_acc_t dwspace;
    idxs_acc_t iwspace;
};

} // namespace impl

template <typename Float>
select_small_k_l2<Float>::select_small_k_l2(cl::sycl::queue& queue)
        : q(queue),
          max_work_group_size(
              queue.get_device().template get_info<cl::sycl::info::device::max_work_group_size>()),
          max_local_size(
              queue.get_device().template get_info<cl::sycl::info::device::local_mem_size>() /
              elem_size),
          preferred_width(queue.get_device().template get_info<preferred_size_flag>()) {}

template <typename Float>
std::pair<std::int64_t, std::int64_t> select_small_k_l2<Float>::preferred_local_size(const std::int64_t k) {
    if (k > max_local_size)
        throw std::exception();
    const idx_t min_k_width    = preferred_size * (k / preferred_size + 1);
    const idx_t yrange_mem_lim = max_local_size / min_k_width;
    const idx_t yrange_max_lim = std::min(yrange_mem_lim, max_work_group_size);
    const idx_t yrange_min_lim =
        std::min(yrange_mem_lim, std::min(prefered_size, max_work_group_size));
    idx_t yrange = yrange_min_lim, k_width = min_k_width;
    for (; yrange < yrange_max_lim; yrange++) {
        // Optimal k_width
        k_width = max_local_size / yrange;
        if (k_width % preffered_size == 0)
            break;
    }
    if (yrange * k_width > max_local_size)
        throw std::exception();
    if (yrange > max_work_group_size)
        throw std::exception();
    return std::pair<std::int64_t, std::int64_t>(k_width, y_range);
}

template <typename Float>
cl::sycl::event select_small_k_l2<Float>::operator()(const Float* cross,
                                                     const Float* norms,
                                                     const std::int64_t batch_size,
                                                     const std::int64_t queue_size,
                                                     const std::int64_t k,
                                                     idx_t* nearest_indices,
                                                     Float* nearest_distances,
                                                     const std::int64_t k_width,
                                                     const std::int64_t yrange) {
    const idx_t local_buff_size = k_width * yrange; 
    if (local_buff_size > max_local_size)
        throw std::exception();
    if (yrange <= 0)
        throw std::exception();
    auto result = this->q.submit([&](cl::sycl::handler& handler) {
        typedef cl::sycl::
            accessor<idx_t, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                local_idxs_acc_t;
        typedef cl::sycl::
            accessor<Float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                local_dist_acc_t;
        size_t xrange = max_work_group_size / yrange;
        cl::sycl::range<2> local_range{ static_cast<size_t>(), static_cast<size_t>(yrange) };
        auto local_indices = local_idxs_acc_t(cl::sycl::range<1>{ local_buff_size }, handler);
        auto local_distances = local_dist_acc_t(cl::sycl::range<1>{ local_buff_size }, handler);
        auto functor_instance =
            kernel_t{ /*.input_cross = */ cross,
                      /*.input_norms = */ norms,
                      /*.output_indices = */ nearest_indices,
                      /*.output_distances = */ nearest_distances,
                      /*.k = */ k,
                      /*.n_vectors = */ queue_size,
                      /*.k_width = */ k_width,
                      /*.vector_size = */ vector_size,
                      /*.dwspace = */ local_distances,
                      /*.iwspace = */ local_indices
            };
        handler.parallel_for<kernel_t>(call_range, functor_instance); 
    });
    return result;
}

template <typename Float>
cl::sycl::event select_small_k_l2<Float>::operator()(const Float* cross,
                                                     const Float* norms,
                                                     const std::int64_t batch_size,
                                                     const std::int64_t queue_size,
                                                     const std::int64_t k,
                                                     idx_t* nearest_indices,
                                                     Float* nearest_distances) {
    auto [k_width, yrange] = this->preferred_local_size(k);
    return this->()(cross,
                    norms,
                    batch_size,
                    queue_size,
                    k,
                    nearest_indices,
                    nearest_distances,
                    k_width,
                    yrange);
}

template struct select_small_k_l2<float>;
template struct select_small_k_l2<double>;

#endif

} // namespace oneapi::dal::backend::primitives
