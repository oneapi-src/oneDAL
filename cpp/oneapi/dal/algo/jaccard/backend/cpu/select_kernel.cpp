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

#include "oneapi/dal/algo/jaccard/backend/cpu/select_kernel.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {
extern similarity_result call_jaccard_block_kernel(const descriptor_base &desc,
                                                   const similarity_input &input);

template <typename Float, typename Method>
similarity_result backend_block<Float, Method>::operator()(const descriptor_base &desc,
                                                           const similarity_input &input) {
    return call_jaccard_block_kernel(desc, input);
}

template <>
std::shared_ptr<backend_base> get_backend<float, method::by_default>(
    const descriptor_base &desc,
    const similarity_input &input) {
    return std::shared_ptr<backend_base>(new backend_block<float, method::by_default>);
}

} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
