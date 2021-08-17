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

#include "oneapi/dal/backend/primitives/voting/uniform.hpp"

namespace oneapi::dal::backend::primitives {

template<typename ClassType>
uniform_voting<ClassType>::uniform_voting(sycl::queue& q)
    : queue_{ q } {}

template<typename ClassType>
sycl::queue& uniform_voting<ClassType>::get_queue() const {
    return this->queue_;
}

template<typename ClassType>
std::unique_ptr<uniform_voting<ClassType>> make_uniform_votiung(
        sycl::queue& queue, std::int64_t max_block, std::int64_t k_response) {
    using small_k = small_k_uniform_voting<ClassType>;
    using large_k = large_k_uniform_voting<ClassType>;
    //if(k_response < small_k::k_max) {
    //
    //}
    return large_k(queue, max_block, k_response);
}

#define INSTANTIATE(CLASS)          \
template class uniform_voting<CLASS>;

INSTANTIATE(std::int32_t);
INSTANTIATE(std::int64_t);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
