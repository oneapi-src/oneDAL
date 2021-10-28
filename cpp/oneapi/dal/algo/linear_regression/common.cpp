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

#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::linear_regression {

namespace detail {

namespace v1 {
template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl() = default;

    bool compute_intercept = true;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
descriptor_base<Task>::descriptor_base(bool compute_intercept)
        : impl_(new descriptor_impl<Task>{}) {
    impl_->compute_intercept = compute_intercept;
}

template <typename Task>
bool descriptor_base<Task>::get_compute_intercept() const {
    return impl_->compute_intercept;
}

template <typename Task>
void descriptor_base<Task>::set_compute_intercept_impl(bool compute_intercept) {
    impl_->compute_intercept = compute_intercept;
}

template class ONEDAL_EXPORT descriptor_base<task::regression>;

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::model_impl;

template <typename Task>
model<Task>::model() : impl_(nullptr) {}

template <typename Task>
model<Task>::model(const std::shared_ptr<detail::model_impl<Task>>& impl) : impl_(impl) {}

template <typename Task>
const table& model<Task>::get_betas() const {
    return impl_->get_betas();
}

template <typename Task>
void model<Task>::serialize(dal::detail::output_archive& ar) const {
    dal::detail::serialize_polymorphic_shared(impl_, ar);
}

template <typename Task>
void model<Task>::deserialize(dal::detail::input_archive& ar) {
    dal::detail::deserialize_polymorphic_shared(impl_, ar);
}

template class ONEDAL_EXPORT model<task::regression>;

ONEDAL_REGISTER_SERIALIZABLE(backend::norm_eq_model_impl<task::regression>)

} // namespace v1
} // namespace oneapi::dal::linear_regression
