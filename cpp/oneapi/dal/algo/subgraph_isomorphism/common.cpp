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

#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {

class detail::descriptor_impl : public base {
public:
    bool semantic_match = false;
    std::int64_t max_match_count = 0;
    kind _kind = kind::induced;
};

using detail::descriptor_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

kind descriptor_base::get_kind() const {
    return impl_->_kind;
}

bool descriptor_base::get_semantic_match() const {
    return impl_->semantic_match;
}

std::int64_t descriptor_base::get_max_match_count() const {
    return impl_->max_match_count;
}

void descriptor_base::set_kind_impl(kind value) {
    impl_->_kind = value;
}

void descriptor_base::set_semantic_match_impl(bool semantic_match) {
    impl_->semantic_match = semantic_match;
}

void descriptor_base::set_max_match_count_impl(std::int64_t max_match_count) {
    impl_->max_match_count = max_match_count;
}

} // namespace subgraph_isomorphism
} // namespace oneapi::dal::preview
