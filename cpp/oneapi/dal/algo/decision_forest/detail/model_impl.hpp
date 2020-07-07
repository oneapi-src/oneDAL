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

namespace oneapi::dal::decision_forest {

template <typename Task>
class detail::model_impl : public base {
public:
    virtual std::int64_t get_tree_count() const {
        return 0;
    }
    virtual std::int64_t get_class_count() const {
        return 0;
    }
    virtual void clear() {}

    virtual bool is_interop() const {
        return false;
    }
};

} // namespace oneapi::dal::decision_forest
