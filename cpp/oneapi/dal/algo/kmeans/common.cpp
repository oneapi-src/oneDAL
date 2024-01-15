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

#include "oneapi/dal/backend/serialization.hpp"
#include "oneapi/dal/algo/kmeans/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::kmeans {
namespace detail {
namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    std::int64_t cluster_count = 2;
    std::int64_t max_iteration_count = 100;
    double accuracy_threshold = 0;
};

template <typename Task>
class model_impl : public ONEDAL_SERIALIZABLE(kmeans_clustering_model_impl_id) {
public:
    void serialize(dal::detail::output_archive& ar) const override {
        ar(centroids);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(centroids);
    }

    table centroids;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_cluster_count() const {
    return impl_->cluster_count;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_iteration_count() const {
    return impl_->max_iteration_count;
}

template <typename Task>
double descriptor_base<Task>::get_accuracy_threshold() const {
    return impl_->accuracy_threshold;
}

template <typename Task>
void descriptor_base<Task>::set_cluster_count_impl(std::int64_t value) {
    if (value <= 0) {
        throw domain_error(dal::detail::error_messages::cluster_count_leq_zero());
    }
    if (value > dal::detail::limits<std::int32_t>::max()) {
        throw domain_error(dal::detail::error_messages::cluster_count_gt_max_int32());
    }
    impl_->cluster_count = value;
}

template <typename Task>
void descriptor_base<Task>::set_max_iteration_count_impl(std::int64_t value) {
    if (value < 0) {
        throw domain_error(dal::detail::error_messages::max_iteration_count_lt_zero());
    }
    impl_->max_iteration_count = value;
}

template <typename Task>
void descriptor_base<Task>::set_accuracy_threshold_impl(double value) {
    if (value < 0.0) {
        throw domain_error(dal::detail::error_messages::accuracy_threshold_lt_zero());
    }
    impl_->accuracy_threshold = value;
}

template class ONEDAL_EXPORT descriptor_base<task::clustering>;

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::model_impl;

template <typename Task>
model<Task>::model() : impl_(new model_impl<Task>{}) {}

template <typename Task>
const table& model<Task>::get_centroids() const {
    return impl_->centroids;
}

template <typename Task>
std::int64_t model<Task>::get_cluster_count() const {
    return impl_->centroids.get_row_count();
}

template <typename Task>
void model<Task>::set_centroids_impl(const table& value) {
    impl_->centroids = value;
}

template <typename Task>
void model<Task>::serialize(dal::detail::output_archive& ar) const {
    dal::detail::serialize_polymorphic_shared(impl_, ar);
}

template <typename Task>
void model<Task>::deserialize(dal::detail::input_archive& ar) {
    dal::detail::deserialize_polymorphic_shared(impl_, ar);
}

template class ONEDAL_EXPORT model<task::clustering>;

ONEDAL_REGISTER_SERIALIZABLE(detail::model_impl<task::clustering>)

} // namespace v1
} // namespace oneapi::dal::kmeans
