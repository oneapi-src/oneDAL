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

#include "oneapi/dal/algo/pca/infer_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::pca {

template <typename Task>
class detail::v1::infer_input_impl : public base {
public:
    infer_input_impl(const model<Task>& trained_model, const table& data)
            : trained_model(trained_model),
              data(data) {}
    model<Task> trained_model;
    table data;
};

template <typename Task>
class detail::v1::infer_result_impl : public base {
public:
    //TODO: disucc naming of components
    table transformed_data;
    table singular_values;
    table explained_variances;
};

using detail::v1::infer_input_impl;
using detail::v1::infer_result_impl;

namespace v1 {

template <typename Task>
infer_input<Task>::infer_input(const model<Task>& trained_model, const table& data)
        : impl_(new infer_input_impl<Task>(trained_model, data)) {}

template <typename Task>
const model<Task>& infer_input<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
const table& infer_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
void infer_input<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
void infer_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
infer_result<Task>::infer_result() : impl_(new infer_result_impl<Task>{}) {}

template <typename Task>
const table& infer_result<Task>::get_transformed_data() const {
    return impl_->transformed_data;
}

template <typename Task>
void infer_result<Task>::set_transformed_data_impl(const table& value) {
    impl_->transformed_data = value;
}

template <typename Task>
const table& infer_result<Task>::get_singular_values() const {
    return impl_->singular_values;
}

template <typename Task>
void infer_result<Task>::set_singular_values_impl(const table& value) {
    impl_->singular_values = value;
}

template <typename Task>
const table& infer_result<Task>::get_explained_variances() const {
    return impl_->explained_variances;
}

template <typename Task>
void infer_result<Task>::set_explained_variances_impl(const table& value) {
    impl_->explained_variances = value;
}

template class ONEDAL_EXPORT infer_input<task::dim_reduction>;
template class ONEDAL_EXPORT infer_result<task::dim_reduction>;

} // namespace v1
} // namespace oneapi::dal::pca
