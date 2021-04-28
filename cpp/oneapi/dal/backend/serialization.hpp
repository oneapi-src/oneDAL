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

#pragma once

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"

#define __ONEDAL_REGISTER_SERIALIZABLE_INIT__(unique_id) \
    namespace oneapi::dal::backend {                     \
    char __force_serializable_init__##unique_id() {      \
        return 0;                                        \
    }                                                    \
    }

#define __ONEDAL_FORCE_SERIALIZABLE_INIT__(unique_id)                                    \
    namespace oneapi::dal::backend {                                                     \
    char __force_serializable_init__##unique_id();                                       \
    }                                                                                    \
    [[maybe_unused]] static volatile char __force_serializable_init_dummy__##unique_id = \
        ::oneapi::dal::backend::__force_serializable_init__##unique_id();

#define __ONEDAL_SERIALIZABLE__(id) \
    ::oneapi::dal::detail::serializable<::oneapi::dal::backend::serialization_ids::id>

namespace oneapi::dal::backend {

#define ID(unique_id, name) static constexpr std::uint64_t name = unique_id

class serialization_ids {
public:
    ID(1000000, array);
    ID(1100000, empty_table_metadata);
    ID(1200000, simple_table_metadata);
    ID(2000000, empty_table);
    ID(2100000, homogen_table);
};

#undef ID

} // namespace oneapi::dal::backend
