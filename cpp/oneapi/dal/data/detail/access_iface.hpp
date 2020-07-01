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

#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::detail {

struct row_block {
    range rows;

    row_block(const range& rows)
        : rows(rows) {}
};

struct column_values_block {
    std::int64_t column_index;
    range rows;

    column_values_block(std::int64_t idx, const range& rows)
        : column_index(idx),
          rows(rows) {}
};

template <typename PolicyType, typename AllocKind>
struct access_iface {
    using array_f32 = array<float>;
    using array_f64 = array<double>;
    using array_i32 = array<std::int32_t>;

    virtual ~access_iface() {}

    virtual void pull(const PolicyType&, array_f32&, const row_block&, const AllocKind&) const = 0;
    virtual void pull(const PolicyType&, array_f64&, const row_block&, const AllocKind&) const = 0;
    virtual void pull(const PolicyType&, array_i32&, const row_block&, const AllocKind&) const = 0;
    virtual void pull(const PolicyType&, array_f32&, const column_values_block&, const AllocKind&) const = 0;
    virtual void pull(const PolicyType&, array_f64&, const column_values_block&, const AllocKind&) const = 0;
    virtual void pull(const PolicyType&, array_i32&, const column_values_block&, const AllocKind&) const = 0;

    virtual void push(const PolicyType&, const array_f32&, const row_block&, const AllocKind&) = 0;
    virtual void push(const PolicyType&, const array_f64&, const row_block&, const AllocKind&) = 0;
    virtual void push(const PolicyType&, const array_i32&, const row_block&, const AllocKind&) = 0;
    virtual void push(const PolicyType&, const array_f32&, const column_values_block&, const AllocKind&) = 0;
    virtual void push(const PolicyType&, const array_f64&, const column_values_block&, const AllocKind&) = 0;
    virtual void push(const PolicyType&, const array_i32&, const column_values_block&, const AllocKind&) = 0;
};

using host_access_iface = access_iface<host_seq_policy, host_only_alloc>;

#ifdef ONEAPI_DAL_DATA_PARALLEL
using dpc_access_iface = access_iface<sycl::queue, sycl::usm::alloc>;
#endif

class access_provider_iface {
public:
    virtual ~access_provider_iface() {}

    virtual host_access_iface& get_host_access_iface() const = 0;
#ifdef ONEAPI_DAL_DATA_PARALLEL
    virtual dpc_access_iface& get_dpc_access_iface() const = 0;
#endif
};

} // namespace oneapi::dal::detail
