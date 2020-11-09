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

#include "oneapi/dal/table/detail/access_iface.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T, typename BlockIndex>
class accessor_base {
public:
    using data_t = std::remove_const_t<T>;

public:
    static constexpr bool is_readonly = std::is_const_v<T>;

#ifdef ONEDAL_DATA_PARALLEL
    template <typename K>
    accessor_base(const K& obj)
            : host_access_(cast_impl<access_provider_iface>(obj).get_access_iface_host()),
              dpc_access_(cast_impl<access_provider_iface>(obj).get_access_iface_dpc()) {}
#else
    template <typename K>
    accessor_base(const K& obj)
            : host_access_(cast_impl<access_provider_iface>(obj).get_access_iface_host()) {}
#endif

    template <typename Policy, typename Allocator>
    array<data_t> pull(const Policy& policy, const BlockIndex& idx, const Allocator& alloc) const {
        array<data_t> block;
        get_access(policy).pull(policy, block, idx, alloc);
        return block;
    }

    template <typename Policy, typename Allocator>
    T* pull(const Policy& policy,
            array<data_t>& block,
            const BlockIndex& idx,
            const Allocator& alloc) const {
        get_access(policy).pull(policy, block, idx, alloc);
        if constexpr (is_readonly) {
            return block.get_data();
        }
        else {
            return block.get_mutable_data();
        }
    }

    template <typename Policy>
    void push(const Policy& policy, const array<data_t>& block, const BlockIndex& idx) {
        get_access(policy).push(policy, block, idx);
    }

private:
    access_iface_host& get_access(const default_host_policy&) {
        return host_access_;
    }

    const access_iface_host& get_access(const default_host_policy&) const {
        return host_access_;
    }

#ifdef ONEDAL_DATA_PARALLEL
    access_iface_dpc& get_access(const data_parallel_policy&) {
        return dpc_access_;
    }

    const access_iface_dpc& get_access(const data_parallel_policy&) const {
        return dpc_access_;
    }
#endif

private:
    access_iface_host& host_access_;
#ifdef ONEDAL_DATA_PARALLEL
    access_iface_dpc& dpc_access_;
#endif
};

} // namespace v1

using v1::accessor_base;

} // namespace oneapi::dal::detail
