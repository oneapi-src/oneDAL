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

#include "oneapi/dal/test/engine/feature_types.hpp"

namespace oneapi::dal::test::engine {

std::int64_t feature_types_impl::get_feature_count() const {
    return ftypes_.get_count();
}

const array<feature_type>& feature_types_impl::get_array() const {
    return ftypes_;
}

std::int64_t feature_types::get_feature_count() const {
    return impl_->get_feature_count();
}

const array<feature_type>& feature_types::get_array() const {
    return impl_->get_array();
}

void feature_types_builder_impl::set_default(feature_type type) {
    feature_types_builder_impl::set(range(0, feature_count_), type);
}

void feature_types_builder_impl::set(std::int64_t idx, feature_type type) {
    feature_types_builder_impl::set(range(idx, idx + 1), type);
}

void feature_types_builder_impl::set(const range& r, feature_type type) {
    const std::int64_t left = r.start_idx;
    const std::int64_t right = r.end_idx;

    ONEDAL_ASSERT(left >= 0);
    ONEDAL_ASSERT(left < right);
    ONEDAL_ASSERT(right <= feature_count_);

    auto ptr = ftypes_.get_mutable_data();
    for (std::int64_t i = left; i < right; ++i) {
        ptr[i] = type;
    }
}

feature_types feature_types_builder_impl::build() const {
    return feature_types(ftypes_);
}

feature_types_builder& feature_types_builder::set_default(feature_type type) {
    impl_->set_default(type);
    return *this;
}

feature_types_builder& feature_types_builder::set(std::int64_t idx, feature_type type) {
    impl_->set(idx, type);
    return *this;
}

feature_types_builder& feature_types_builder::set(const range& r, feature_type type) {
    impl_->set(r, type);
    return *this;
}

feature_types feature_types_builder::build() const {
    return impl_->build();
}

} // namespace oneapi::dal::test::engine
