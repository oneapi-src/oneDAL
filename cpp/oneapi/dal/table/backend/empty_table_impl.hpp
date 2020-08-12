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

#include "oneapi/dal/table/table_metadata.hpp"

namespace oneapi::dal::backend {

class empty_table_impl {
public:
    static constexpr std::int64_t pure_empty_table_kind = 0;

public:
    std::int64_t get_column_count() const {
        return 0;
    }

    std::int64_t get_row_count() const {
        return 0;
    }

    std::int64_t get_kind() const {
        return pure_empty_table_kind;
    }

    const table_metadata& get_metadata() const {
        static table_metadata tm;
        return tm;
    }

    template <typename T>
    void pull_rows(array<T>& block, const range&) const {
        block.reset();
    }

    template <typename T>
    void pull_column(array<T>& block, std::int64_t, const range&) const {
        block.reset();
    }
};

} // namespace oneapi::dal::backend
