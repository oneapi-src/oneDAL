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

#include "oneapi/dal/backend/primitives/reduction/common.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"
#include "oneapi/dal/backend/primitives/super_accumulator/super_accumulator.hpp"

namespace oneapi::dal::backend::primitives {

template<typename UnaryOp, int f, int b>
class reduction_kernel {
    using super_accums = super_accumulators<float, false>;

public:
    constexpr static inline float zero = 0.f;
    constexpr static inline int folding = f;
    constexpr static inline int block = b;

    reduction_kernel(std::int32_t width,
                     std::int64_t stride,
                     std::int32_t height, 
                     const float* data, 
                     std::int64_t* bins,
                     const UnaryOp& unary)
        : width_(width),
          stride_(stride),
          height_(height),
          data_(data),
          bins_(bins),
          unary_(unary) {}

    void operator() (sycl::nd_item<2> it) const {
	    // Acumulators for working in width
        float accs[folding] = { zero };
        //
        const int vid = it.get_global_id(1);
        const int hid = it.get_global_id(0);
        const int hwg = it.get_global_range(0);
        //
	    // Reduction Section
	    //
        for(int i = 0; i < block; ++i) {
	        // Current dataset row
            const int rid = vid * block + i;
            for(int j = 0; j < folding; ++j) {
		        // Current dataset col number
                const int cid = hid + j * hwg;
		        // Check for row and col to be in dataset
                const bool val = (rid < height) && (cid < width);
		        // Access to the value in row-major order
                // All arithmetics should work in std::int64_t
                accs[j] += val ? data[cid + rid * stride] : zero;
	        }
        }
	    //
        // Super counter Section
        //
        for(int j = 0; j < folding; ++j) {
            const int cid = hid + j * hwg;
            if(cid < width) {
                bins.add(accs[j], cid);
            }
        }
    }

private:
    const std::int32_t width_;
    const std::int64_t stride_;
    const std::int32_t height_;
    const float* const data_;
    const super_accums bins_;
    const UnaryOp unary_;
};

template<typename UnaryOp, int folding, int block_size>
sycl::event reduction_impl(sycl::queue& queue,
                           const float* data,
                           const std::int32_t width,
                           const std::int64_t stride,
                           const std::int32_t height,
                           std::int64_t* bins,
                           const UnaryOp& unary,
                           const std::vector<sycl::event>& deps = {}) {
    using kernel_t = reduction_kernel<UnaryOp, folding>;
        constexpr int bl = kernel_t::block;
        const int n_blocks = height / bl + bool(height % bl);
        return queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for<kernel_t>(
                sycl::nd_range<2>{
                    sycl::range<2>(wg, n_blocks), 
                    sycl::range<2>(wg, 1)           },
                kernel_t(width, height, data, bins));
        }); 
    }
    return reduction__imp<folding - 1>(queue, 
                                data, 
                                width, 
                                height, 
                                bins,
                                deps);
}

