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
#include <cmath>
#include <exception>
#include <limits>

#ifdef ONEAPI_DAL_DATA_PARALLEL
    #include <CL/sycl.hpp>
#endif

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
                           idx_t k_width_,
                           idx_t vector_size_;
                           dist_acc_t dwspace_, idxs_acc_t iwspace_)
            : input_cross(input_cross_),
              input_norms(input_norms_),
              output_indices(output_indices_),
              output_distances(output_distances_),
              k(k_),
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
            const idx_t* const irow = &(output_indices[gyid * k]);
            const Float* const drow = &(ouput_distances[gyid * k]);
            for (idx_t shift = lxid; shift < k; shift += xrange) {
                iwspace[lrid + shift] = irow[shift];
                dwspace[lrid + shift] = drow[shift];
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
                    // Temporary variables for swap
                    Float tdst;
                    idx_t tidx;
                    // Sorting itself
                    for (idx_t step = 0; step < k_width; ++step) {
                        for (idx_t i = lxid + step % 2; i + 1 < k_width; i += xrange) {
                            if (dwspace[lrid + i] > dwspace[lrid + i + 1]) {
                                // TODO: Another swaping?
                                {
                                    tdst                  = dwspace[lrid + i];
                                    dwspace[lrid + i]     = dwspace[lrid + i + 1];
                                    dwspace[lrid + i + 1] = tdst;
                                }
                                {
                                    tidx                  = iwspace[lrid + i];
                                    iwspace[lrid + i]     = iwspace[lrid + i + 1];
                                    iwspace[lrid + i + 1] = tidx;
                                }
                            }
                        }
                        idx.barrier(cl::sycl::access::fence_space::local_space);
                    }
                }
            }
        }

        //Exit from algorithm
        {
            const idx_t* const irow = &(output_indices[gyid * k]);
            const Float* const drow = &(ouput_distances[gyid * k]);
            for (idx_t shift = lxid; shift < k; shift += xrange) {
                irow[shift] = iwspace[lyid * k_width + shift];
                drow[shift] = dwspace[lyid * k_width + shift];
            }
            // TODO: Optimize with async_work_group_copy
        }
    }

private:
    const Float* const input_cross;
    const Float* const input_norms;
    idx_t* const output_indices;
    Float* const output_distances;
    const idx_t k, k_width, vector_size;
    dist_acc_t dwspace;
    idxs_acc_t iwspace;
};

} // namespace impl

#endif

} // namespace oneapi::dal::backend::primitives