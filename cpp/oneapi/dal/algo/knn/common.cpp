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

#include "oneapi/dal/algo/knn/common.hpp"
#include "oneapi/dal/algo/knn/detail/model_impl.hpp"

namespace oneapi::dal::knn {

class detail::descriptor_impl : public base {
  public:
    std::int64_t class_count = 2;
    std::int64_t neighbor_count = 1;
    std::int64_t seed = 1;
    bool data_use_in_model = false;
};

using detail::descriptor_impl;
using detail::model_impl;

descriptor_base::descriptor_base()
    : impl_(new descriptor_impl{}) {}

std::int64_t descriptor_base::get_class_count() const {
    return impl_->class_count;
}

std::int64_t descriptor_base::get_neighbor_count() const {
    return impl_->neighbor_count;
}

std::int64_t descriptor_base::get_seed() const {
    return impl_->seed;
}

bool descriptor_base::get_data_use_in_model() const {
    return impl_->data_use_in_model;
}

void descriptor_base::set_class_count_impl(std::int64_t value) {
    impl_->class_count = value;
}

void descriptor_base::set_neighbor_count_impl(std::int64_t value) {
    impl_->neighbor_count = value;
}

void descriptor_base::set_seed_impl(std::int64_t value) {
    impl_->seed = value;
}

void descriptor_base::set_data_use_in_model_impl(bool value) {
    impl_->data_use_in_model = value;
}

model::model() : impl_(new model_impl{}) {}

} // namespace oneapi::dal::pca
