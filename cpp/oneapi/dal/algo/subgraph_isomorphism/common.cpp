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

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <typename Task>
class descriptor_impl : public base {
public:
    bool semantic_match = false;
    std::int64_t max_match_count = 0;
    kind _kind = kind::induced;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
kind descriptor_base<Task>::get_kind() const {
    return impl_->_kind;
}

template <typename Task>
bool descriptor_base<Task>::get_semantic_match() const {
    return impl_->semantic_match;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_match_count() const {
    return impl_->max_match_count;
}

template <typename Task>
void descriptor_base<Task>::set_kind(kind kind) {
    impl_->_kind = kind;
}

template <typename Task>
void descriptor_base<Task>::set_semantic_match(bool semantic_match) {
    impl_->semantic_match = semantic_match;
}

template <typename Task>
void descriptor_base<Task>::set_max_match_count(std::int64_t max_match_count) {
    impl_->max_match_count = max_match_count;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
