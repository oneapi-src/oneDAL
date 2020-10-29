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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_interop.hpp"

namespace oneapi::dal::decision_forest {

template <typename Task>
class detail::v1::model_impl : public base {
public:
    model_impl() = default;
    model_impl(const model_impl&) = delete;
    model_impl& operator=(const model_impl&) = delete;

    model_impl(backend::model_interop* interop)
        : interop_(interop) {}

    virtual ~model_impl() {
        delete interop_;
        interop_ = nullptr;
    }

    backend::model_interop* get_interop() const {
        return interop_;
    }

    std::int64_t tree_count = 0;
    std::int64_t class_count = 0;

private:
    backend::model_interop* interop_ = nullptr;
};

using model_impl_cls = detail::model_impl<task::classification>;
using model_impl_reg = detail::model_impl<task::regression>;

} // namespace oneapi::dal::decision_forest
