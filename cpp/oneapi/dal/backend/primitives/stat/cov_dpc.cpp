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

#include "oneapi/dal/backend/primitives/stat/cov.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_cov_op<Float>::operator()(sycl::queue& queue,
                                              const ndview<Float, 2>& data,
                                              const ndview<Float, 1>& sums,
                                              const ndview<Float, 2>& cov,
                                              const event_vector& deps) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(cov.has_mutable_data());
    ONEDAL_ASSERT(data.get_shape(1) == sums.get_shape(0));
    ONEDAL_ASSERT(cov.get_shape(0) == cov.get_shape(1));
    ONEDAL_ASSERT(cov.get_shape(0) == data.get_shape(1));

    // const Float* data_ptr = data.get_data();
    // const Float* means_ptr = means.get_data();

    auto mm_event = gemm(queue, data.t(), data, cov, deps);

    Float* cov_ptr = cov.get_mutable_data();

    const sycl::range<1> range = dal::detail::integral_cast<std::size_t>(cov.get_count());
    return queue.parallel_for(range, { mm_event }, [=](sycl::nd_item<1> id) {
        const std::size_t i = id.get_global_id();
        cov_ptr[i] = Float(0);
    });
}

template struct compute_cov_op<float>;
template struct compute_cov_op<double>;

} // namespace oneapi::dal::backend::primitives
