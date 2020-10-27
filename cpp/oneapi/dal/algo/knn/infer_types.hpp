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

#pragma once

#include "oneapi/dal/algo/knn/common.hpp"

namespace oneapi::dal::knn {

namespace detail {
template <typename Task = task::by_default>
class infer_input_impl;

template <typename Task = task::by_default>
class infer_result_impl;
} // namespace detail

template <typename Task = task::by_default>
class ONEDAL_EXPORT infer_input : public base {
public:
    infer_input(const table& data, const model<Task>& model);

    const table& get_data() const;
    const model<Task>& get_model() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    auto& set_model(const model<Task>& m) {
        set_model_impl(m);
        return *this;
    }

private:
    void set_data_impl(const table& data);
    void set_model_impl(const model<Task>& model);

    dal::detail::pimpl<detail::infer_input_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class ONEDAL_EXPORT infer_result {
public:
    infer_result();

    const table& get_labels() const;

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

private:
    void set_labels_impl(const table&);
    const table& get_labels_impl() const;
    dal::detail::pimpl<detail::infer_result_impl<Task>> impl_;
};

} // namespace oneapi::dal::knn
