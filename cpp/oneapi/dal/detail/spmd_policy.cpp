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

#include "oneapi/dal/detail/spmd_policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

namespace ps = oneapi::dal::preview::spmd;

template<typename memory_access_kind>
class spmd_policy_impl {
public:
    explicit spmd_policy_impl(const ps::communicator<memory_access_kind>& comm) : comm(comm) {}
    ps::communicator<memory_access_kind> comm;
};

template<typename memory_access_kind>
spmd_policy_base<memory_access_kind>::spmd_policy_base(const ps::communicator<memory_access_kind>& comm)
        : impl_(new spmd_policy_impl<memory_access_kind>{ comm }) {}

template<typename memory_access_kind>
const ps::communicator<memory_access_kind>& spmd_policy_base<memory_access_kind>::get_communicator() const {
    return impl_->comm;
}

template class spmd_policy_base<ps::device_memory_access::usm>;
template class spmd_policy_base<ps::device_memory_access::none>;

} // namespace v1
} // namespace oneapi::dal::detail
