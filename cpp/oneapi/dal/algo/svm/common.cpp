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

#include "oneapi/dal/algo/svm/common.hpp"

namespace oneapi::dal::svm {

class detail::descriptor_impl : public base {
public:
    double c                         = 1.0;
    double accuracy_threshold        = 0.001;
    std::int64_t max_iteration_count = 100000;
    double cache_size                = 200.0;
    double tau                       = 1e-6;
    bool shrinking                   = true;
};

class detail::model_impl : public base {
public:
    table support_vectors;
    table coefficients;
    double bias;
    std::int64_t support_vectors_count;
};

using detail::descriptor_impl;
using detail::model_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

double descriptor_base::get_c() const {
    return impl_->c;
}

double descriptor_base::get_accuracy_threshold() const {
    return impl_->accuracy_threshold;
}

std::int64_t descriptor_base::get_max_iteration_count() const {
    return impl_->max_iteration_count;
}

double descriptor_base::get_cache_size() const {
    return impl_->cache_size;
}

double descriptor_base::get_tau() const {
    return impl_->tau;
}

bool descriptor_base::get_shrinking() const {
    return impl_->shrinking;
}

void descriptor_base::set_c_impl(const double value) {
    impl_->c = value;
}

void descriptor_base::set_accuracy_threshold_impl(const double value) {
    impl_->accuracy_threshold = value;
}

void descriptor_base::set_max_iteration_count_impl(const std::int64_t value) {
    impl_->max_iteration_count = value;
}

void descriptor_base::set_cache_size_impl(const double value) {
    impl_->cache_size = value;
}

void descriptor_base::set_tau_impl(const double value) {
    impl_->tau = value;
}

void descriptor_base::set_shrinking_impl(const bool value) {
    impl_->shrinking = value;
}

model::model() : impl_(new model_impl{}) {}

table model::get_support_vectors() const {
    return impl_->support_vectors;
}

table model::get_coefficients() const {
    return impl_->coefficients;
}

double model::get_bias() const {
    return impl_->bias;
}

std::int64_t model::get_support_vector_count() const {
    return impl_->support_vectors_count;
}

void model::set_support_vectors_impl(const table &value) {
    impl_->support_vectors = value;
}

void model::set_coefficients_impl(const table &value) {
    impl_->coefficients = value;
}

void model::set_bias_impl(const double value) {
    impl_->bias = value;
}

void model::set_support_vector_count_impl(const std::int64_t value) {
    impl_->support_vectors_count = value;
}

} // namespace oneapi::dal::svm
