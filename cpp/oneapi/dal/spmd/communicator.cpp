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

#include "oneapi/dal/spmd/communicator.hpp"
#include "oneapi/dal/spmd/detail/communicator_utils.hpp"

namespace oneapi::dal::preview::spmd {

template <typename memory_access_kind>
template <typename D>
request communicator<memory_access_kind>::bcast(const array<D>& ary, std::int64_t root) const {
    return de::bcast(*this, ary, root);
}

template <typename memory_access_kind>
template <typename D>
request communicator<memory_access_kind>::allgather(const array<D>& send,
                                                    const array<D>& recv) const {
    return de::allgather(*this, send, recv);
}

template <typename memory_access_kind>
template <typename D>
request communicator<memory_access_kind>::allgather(D& scalar, const array<D>& recv) const {
    return de::allgather(*this, scalar, recv);
}

template <typename memory_access_kind>
template <typename D>
request communicator<memory_access_kind>::allreduce(const array<D>& ary,
                                                    const reduce_op& op) const {
    return de::allreduce(*this, ary, op);
}

#define INSTANTIATE(M, D)                                                                      \
    template request communicator<M>::bcast(const array<D>& ary, std::int64_t root) const;     \
    template request communicator<M>::allgather<D>(const array<D>& send, const array<D>& recv) \
        const;                                                                                 \
    template request communicator<M>::allgather<D>(D & scalar, const array<D>& recv) const;    \
    template request communicator<M>::allreduce<D>(const array<D>& ary, const reduce_op& op) const;

#define INSTANTIATE_MEMORY_ACCESS(M) \
    INSTANTIATE(M, float)            \
    INSTANTIATE(M, double)           \
    INSTANTIATE(M, std::int8_t)      \
    INSTANTIATE(M, std::int16_t)     \
    INSTANTIATE(M, std::int32_t)     \
    INSTANTIATE(M, std::int64_t)     \
    INSTANTIATE(M, std::uint8_t)     \
    INSTANTIATE(M, std::uint16_t)    \
    INSTANTIATE(M, std::uint32_t)    \
    INSTANTIATE(M, std::uint64_t)

INSTANTIATE_MEMORY_ACCESS(device_memory_access::none)
INSTANTIATE_MEMORY_ACCESS(device_memory_access::usm)

} // namespace oneapi::dal::preview::spmd
