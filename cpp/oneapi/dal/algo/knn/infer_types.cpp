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

#include "oneapi/dal/algo/knn/infer_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::knn {

template <typename Task>
class detail::infer_input_impl : public base {
public:
    infer_input_impl(const table& data, const model<Task>& m) : data(data), trained_model(m) {}

    table data;
    model<Task> trained_model;
};

template <typename Task>
class detail::infer_result_impl : public base {
public:
    table responses;
    table indices;
    table distances;
    result_option_id options;
};

using detail::infer_input_impl;
using detail::infer_result_impl;

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
const table& infer_result<Task>::get_responses() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::responses)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->responses;
}

template <typename Task>
void infer_result<Task>::set_responses_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::responses)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->responses = value;
}

template <typename Task>
const table& infer_result<Task>::get_indices() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::indices)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->indices;
}

template <typename Task>
void infer_result<Task>::set_indices_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::indices)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->indices = value;
}

template <typename Task>
const table& infer_result<Task>::get_distances() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::distances)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->distances;
}

template <typename Task>
void infer_result<Task>::set_distances_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::distances)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->distances = value;
}

template <typename Task>
const result_option_id& infer_result<Task>::get_result_options() const {
    return impl_->options;
}

template <typename Task>
void infer_result<Task>::set_result_options_impl(const result_option_id& value) {
    impl_->options = value;
}

template class ONEDAL_EXPORT infer_input<task::classification>;
template class ONEDAL_EXPORT infer_result<task::classification>;
template class ONEDAL_EXPORT infer_input<task::regression>;
template class ONEDAL_EXPORT infer_result<task::regression>;
template class ONEDAL_EXPORT infer_input<task::search>;
template class ONEDAL_EXPORT infer_result<task::search>;

} // namespace oneapi::dal::knn
