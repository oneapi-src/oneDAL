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

#include "oneapi/dal/algo/pca/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::pca {

class detail::descriptor_impl : public base {
public:
    std::int64_t component_count = -1;
    bool is_deterministic = false;
};

class detail::model_impl : public base {
public:
    table eigenvectors;
};

using detail::descriptor_impl;
using detail::model_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

std::int64_t descriptor_base::get_component_count() const {
    return impl_->component_count;
}

bool descriptor_base::get_is_deterministic() const {
    return impl_->is_deterministic;
}

void descriptor_base::set_component_count_impl(std::int64_t value) {
    if (value < 0) {
        throw domain_error("Descriptor component_count should be >= 0");
    }
    impl_->component_count = value;
}

void descriptor_base::set_is_deterministic_impl(bool value) {
    impl_->is_deterministic = value;
}

model::model() : impl_(new model_impl{}) {}

table model::get_eigenvectors() const {
    return impl_->eigenvectors;
}

void model::set_eigenvectors_impl(const table& value) {
    impl_->eigenvectors = value;
}

} // namespace oneapi::dal::pca
