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

namespace oneapi::dal::knn {

class detail::infer_input_impl : public base {
public:
    infer_input_impl(const table& data, const model& m) : data(data), trained_model(m) {}

    table data;
    model trained_model;
};

class detail::infer_result_impl : public base {
public:
    table labels;
};

using detail::infer_input_impl;
using detail::infer_result_impl;

infer_input::infer_input(const table& data, const model& m)
        : impl_(new infer_input_impl(data, m)) {}

table infer_input::get_data() const {
    return impl_->data;
}

model infer_input::get_model() const {
    return impl_->trained_model;
}

void infer_input::set_data_impl(const table& value) {
    impl_->data = value;
}

void infer_input::set_model_impl(const model& value) {
    impl_->trained_model = value;
}

infer_result::infer_result() : impl_(new infer_result_impl{}) {}

const table& infer_result::get_labels() const {
    return impl_->labels;
}

void infer_result::set_labels_impl(const table& value) {
    impl_->labels = value;
}

} // namespace oneapi::dal::knn
