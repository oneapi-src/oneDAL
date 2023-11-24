/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/backend/primitives/sign_flip/sign_flip.hpp"
#include "oneapi/dal/backend/primitives/loops.hpp"
#include <sycl/ext/oneapi/experimental/builtins.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
inline sycl::event sign_flip_impl(sycl::queue q,
                                  Float* eigvecs,
                                  std::int64_t row_count,
                                  std::int64_t column_count,
                                  const event_vector& deps) {
    ONEDAL_ASSERT(eigvecs);
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, row_count, column_count);

    auto update_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(row_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];

            std::int64_t max_index = 0;
            for (std::int64_t j = 1; j < column_count; ++j) {
                if (std::fabs(eigvecs[i * column_count + j]) >
                    std::fabs(eigvecs[i * column_count + max_index])) {
                    max_index = j;
                }
            }
            if (eigvecs[i * column_count + max_index] < 0) {
                for (std::int64_t j = 0; j < column_count; ++j) {
                    eigvecs[i * column_count + j] = -eigvecs[i * column_count + j];
                }
            }
        });
    });

    return update_event;
}

template <typename Float>
sycl::event sign_flip(sycl::queue& q, ndview<Float, 2>& eigvecs, const event_vector& deps) {
    auto event = sign_flip_impl(q,
                                eigvecs.get_mutable_data(),
                                eigvecs.get_dimension(0),
                                eigvecs.get_dimension(1),
                                deps);
    return event;
}

#define INSTANTIATE_SIGN_FLIP(Float)                                       \
    template ONEDAL_EXPORT sycl::event sign_flip<Float>(sycl::queue&,      \
                                                        ndview<Float, 2>&, \
                                                        const event_vector&);

INSTANTIATE_SIGN_FLIP(float)
INSTANTIATE_SIGN_FLIP(double)

} // namespace oneapi::dal::backend::primitives
