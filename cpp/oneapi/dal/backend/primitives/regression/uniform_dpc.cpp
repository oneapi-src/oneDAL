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

#include <algorithm>

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/regression/uniform.hpp"

namespace oneapi::dal::backend::primitives {

template <typename ResponseType>
uniform_regression<ResponseType>::uniform_regression(sycl::queue& q) : queue_{ q } {}

template <typename ResponseType>
uniform_regression<ResponseType>::~uniform_regression() {
    get_queue().wait_and_throw();
}

template <typename ResponseType>
sycl::queue& uniform_regression<ResponseType>::get_queue() const {
    return this->queue_;
}

template <typename ResponseType>
std::unique_ptr<uniform_regression<ResponseType>> make_uniform_regression(sycl::queue& queue,
                                                                          std::int64_t max_block,
                                                                          std::int64_t k_response) {
    using naive = naive_uniform_regression<ResponseType>;

    return std::make_unique<naive>(queue);
}

#define INSTANTIATE(RESPONSE)                                                                 \
    template class uniform_regression<RESPONSE>;                                              \
    template std::unique_ptr<uniform_regression<RESPONSE>> make_uniform_regression<RESPONSE>( \
        sycl::queue&,                                                                         \
        std::int64_t,                                                                         \
        std::int64_t);

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
