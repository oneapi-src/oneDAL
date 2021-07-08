/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::pca {
namespace detail {
namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    std::int64_t component_count = -1;
    bool deterministic = false;
};

template <typename Task>
class model_impl : public ONEDAL_SERIALIZABLE(pca_dim_reduction_model_impl_id) {
public:
    table eigenvectors;

    void serialize(dal::detail::output_archive& ar) const override {
        ar(eigenvectors);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(eigenvectors);
    }
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_component_count() const {
    return impl_->component_count;
}

template <typename Task>
bool descriptor_base<Task>::get_deterministic() const {
    return impl_->deterministic;
}

template <typename Task>
void descriptor_base<Task>::set_component_count_impl(std::int64_t value) {
    if (value < 0) {
        throw domain_error(dal::detail::error_messages::component_count_lt_zero());
    }
    impl_->component_count = value;
}

template <typename Task>
void descriptor_base<Task>::set_deterministic_impl(bool value) {
    impl_->deterministic = value;
}

template class ONEDAL_EXPORT descriptor_base<task::dim_reduction>;

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::model_impl;

template <typename Task>
model<Task>::model() : impl_(new model_impl<Task>{}) {}

template <typename Task>
const table& model<Task>::get_eigenvectors() const {
    return impl_->eigenvectors;
}

template <typename Task>
void model<Task>::set_eigenvectors_impl(const table& value) {
    impl_->eigenvectors = value;
}

template <typename Task>
void model<Task>::serialize(dal::detail::output_archive& ar) const {
    dal::detail::serialize_polymorphic_shared(impl_, ar);
}

template <typename Task>
void model<Task>::deserialize(dal::detail::input_archive& ar) {
    dal::detail::deserialize_polymorphic_shared(impl_, ar);
}

template class ONEDAL_EXPORT model<task::dim_reduction>;
ONEDAL_REGISTER_SERIALIZABLE(model_impl<task::dim_reduction>)

} // namespace v1
} // namespace oneapi::dal::pca
