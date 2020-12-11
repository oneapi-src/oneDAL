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

#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/algo/jaccard/common.hpp"

namespace oneapi::dal::preview {
namespace jaccard {

class detail::vertex_similarity_result_impl : public base {
public:
    table coeffs;
    table vertex_pairs;
    std::int64_t nonzero_coeff_count;
};

using detail::vertex_similarity_result_impl;

vertex_similarity_result::vertex_similarity_result() : impl_(new vertex_similarity_result_impl()) {}

vertex_similarity_result::vertex_similarity_result(const table& vertex_pairs,
                                                   const table& coeffs,
                                                   std::int64_t nonzero_coeff_count)
        : impl_(new vertex_similarity_result_impl()) {
    impl_->vertex_pairs = vertex_pairs;
    impl_->coeffs = coeffs;
    impl_->nonzero_coeff_count = nonzero_coeff_count;
}

table vertex_similarity_result::get_coeffs() const {
    return impl_->coeffs;
}

table vertex_similarity_result::get_vertex_pairs() const {
    return impl_->vertex_pairs;
}

int64_t vertex_similarity_result::get_nonzero_coeff_count() const {
    return impl_->nonzero_coeff_count;
}
} // namespace jaccard
} // namespace oneapi::dal::preview