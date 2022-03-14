/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/pca/common.hpp"

namespace oneapi::dal::pca::backend {

template <typename Descriptor>
inline std::int64_t get_component_count(const Descriptor& desc, const table& data) {
    ONEDAL_ASSERT(desc.get_component_count() >= 0);
    if (desc.get_component_count() == 0) {
        return data.get_column_count();
    }
    return desc.get_component_count();
}

} // namespace oneapi::dal::pca::backend
