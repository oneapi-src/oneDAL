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

#include "oneapi/dal/algo/decision_forest/infer_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest {

template <typename Task>
class detail::infer_input_impl : public base {
public:
    infer_input_impl(const model<Task>& trained_model,
                     const table& data,
                     std::uint64_t results_to_compute)
            : trained_model(trained_model),
              data(data),
              results_to_compute(results_to_compute) {}
    infer_input_impl(
        const model<Task>& trained_model,
        const table& data,
        infer_result_to_compute results_to_compute = infer_result_to_compute::compute_class_labels)
            : trained_model(trained_model),
              data(data),
              results_to_compute(static_cast<std::uint64_t>(results_to_compute)) {}
    model<Task> trained_model;
    table data;
    std::uint64_t results_to_compute =
        static_cast<std::uint64_t>(infer_result_to_compute::compute_class_labels);
};

template <typename Task>
class detail::infer_result_impl : public base {
public:
    table labels;
    table probabilities;
};

using detail::infer_input_impl;
using detail::infer_result_impl;

template <typename Task>
infer_input<Task>::infer_input(const model<Task>& trained_model,
                               const table& data,
                               std::uint64_t results_to_compute)
        : impl_(new infer_input_impl<Task>(trained_model, data, results_to_compute)) {}

template <typename Task>
infer_input<Task>::infer_input(const model<Task>& trained_model,
                               const table& data,
                               infer_result_to_compute results_to_compute)
        : impl_(new infer_input_impl<Task>(trained_model, data, results_to_compute)) {}

template <typename Task>
model<Task> infer_input<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
table infer_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
std::uint64_t infer_input<Task>::get_results_to_compute() const {
    return impl_->results_to_compute;
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
void infer_input<Task>::set_results_to_compute_impl(std::uint64_t value) {
    impl_->results_to_compute = value;
}

template class ONEAPI_DAL_EXPORT infer_input<task::classification>;
template class ONEAPI_DAL_EXPORT infer_input<task::regression>;

/* infer_result implementation */

template <typename Task>
infer_result<Task>::infer_result() : impl_(new infer_result_impl<Task>{}) {}

template <typename Task>
table infer_result<Task>::get_labels() const {
    return impl_->labels;
}

template <typename Task>
table infer_result<Task>::get_probabilities_impl() const {
    return impl_->probabilities;
}

template <typename Task>
void infer_result<Task>::set_labels_impl(const table& value) {
    impl_->labels = value;
}

template <typename Task>
void infer_result<Task>::set_probabilities_impl(const table& value) {
    impl_->probabilities = value;
}

template class ONEAPI_DAL_EXPORT infer_result<task::classification>;
template class ONEAPI_DAL_EXPORT infer_result<task::regression>;

} // namespace oneapi::dal::decision_forest
