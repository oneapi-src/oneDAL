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

#include "oneapi/dal/algo/kmeans/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::kmeans {

class detail::descriptor_impl : public base {
public:
    std::int64_t cluster_count = 2;
    std::int64_t max_iteration_count = 100;
    double accuracy_threshold = 0;
};

class detail::model_impl : public base {
public:
    table centroids;
};

using detail::descriptor_impl;
using detail::model_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

std::int64_t descriptor_base::get_cluster_count() const {
    return impl_->cluster_count;
}

std::int64_t descriptor_base::get_max_iteration_count() const {
    return impl_->max_iteration_count;
}

double descriptor_base::get_accuracy_threshold() const {
    return impl_->accuracy_threshold;
}

void descriptor_base::set_cluster_count_impl(std::int64_t value) {
    if (value <= 0) {
        throw domain_error("cluster_count should be > 0");
    }
    impl_->cluster_count = value;
}

void descriptor_base::set_max_iteration_count_impl(std::int64_t value) {
    if (value < 0) {
        throw domain_error("max_iteration_count should be >= 0");
    }
    impl_->max_iteration_count = value;
}

void descriptor_base::set_accuracy_threshold_impl(double value) {
    if (value < 0.0) {
        throw domain_error("accuracy_threshold should be >= 0.0");
    }
    impl_->accuracy_threshold = value;
}

model::model() : impl_(new model_impl{}) {}

table model::get_centroids() const {
    return impl_->centroids;
}

std::int64_t model::get_cluster_count() const {
    return impl_->centroids.get_row_count();
}

void model::set_centroids_impl(const table& value) {
    impl_->centroids = value;
}

} // namespace oneapi::dal::kmeans
