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

#include "oneapi/dal/algo/knn/infer_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::knn {

template <typename Task>
class detail::v1::infer_input_impl : public base {
public:
    infer_input_impl(const table& data, const model<Task>& m) : data(data), trained_model(m) {}

    table data;
    model<Task> trained_model;
};

template <typename Task>
class detail::v1::infer_result_impl : public base {
public:
    table labels;
    table indices;
    table distances;
};

using detail::v1::infer_input_impl;
using detail::v1::infer_result_impl;

namespace v1 {

template <typename Task>
infer_input<Task>::infer_input(const table& data, const model<Task>& m)
        : impl_(new infer_input_impl<Task>(data, m)) {}

template <typename Task>
const table& infer_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
const model<Task>& infer_input<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
void infer_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void infer_input<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
infer_result<Task>::infer_result() : impl_(new infer_result_impl<Task>{}) {}

template <typename Task>
const table& infer_result<Task>::get_labels() const {
    return impl_->labels;
}

template <typename Task>
void infer_result<Task>::set_labels_impl(const table& value) {
    impl_->labels = value;
}

template <typename Task>
const table& infer_result<Task>::get_indices() const {
    return impl_->indices;
}

template <typename Task>
void infer_result<Task>::set_indices_impl(const table& value) {
    impl_->indices = value;
}

template <typename Task>
const table& infer_result<Task>::get_distances() const {
    return impl_->distances;
}

template <typename Task>
void infer_result<Task>::set_distances_impl(const table& value) {
    impl_->distances = value;
}

template class ONEDAL_EXPORT infer_input<task::classification>;
template class ONEDAL_EXPORT infer_result<task::classification>;
template class ONEDAL_EXPORT infer_input<task::search>;
template class ONEDAL_EXPORT infer_result<task::search>;

} // namespace v1
} // namespace oneapi::dal::knn
