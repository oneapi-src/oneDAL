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

#include <numeric>
#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/communicator_iface.hpp"

namespace oneapi::dal::detail {
namespace v1 {

// TODO: Support other reduction operations, now only SUM available


// TODO: Support other reduction operations, now only SUM available
template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline spmd_request allreduce(const dal::detail::spmd_communicator& comm,
                              const dal::array<T>& ary) {
    ONEDAL_ASSERT(ary.get_count() > 0);
    ONEDAL_ASSERT(ary.has_mutable_data());

#ifdef ONEDAL_ENABLE_ASSERT
    const auto counts = gather(comm, ary.get_count());
    if (is_root_rank(comm)) {
        ONEDAL_ASSERT(counts.size() > 0);
        for (const auto& count : counts) {
            ONEDAL_ASSERT(count == counts.front());
        }
    }
#endif

    spmd_request request;

    __ONEDAL_IF_QUEUE__(ary.get_queue(), {
        auto q = ary.get_queue().value();
        request = comm.allreduce(q,
                                 reinterpret_cast<const byte_t*>(ary.get_data()),
                                 reinterpret_cast<byte_t*>(ary.get_mutable_data()),
                                 ary.get_size(),
                                 dal::detail::make_data_type<T>(),
                                 spmd_reduce_op::sum);
    });

    __ONEDAL_IF_NO_QUEUE__(ary.get_queue(), {
        request = comm.allreduce(reinterpret_cast<const byte_t*>(ary.get_data()),
                                 reinterpret_cast<byte_t*>(ary.get_mutable_data()),
                                 ary.get_size(),
                                 dal::detail::make_data_type<T>(),
                                 spmd_reduce_op::sum);
    });

    return request;
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline spmd_request allgather(const dal::detail::spmd_communicator& comm,
                              const dal::array<T>& send_ary,
                              const dal::array<T>& recv_ary) {
    ONEDAL_ASSERT(send_ary.get_count() > 0);
    ONEDAL_ASSERT(recv_ary.has_mutable_data());

#ifdef ONEDAL_ENABLE_ASSERT
    const auto counts = gather(comm, send_ary.get_count());
    if (is_root_rank(comm)) {
        ONEDAL_ASSERT(counts.size() > 0);
        for (const auto& count : counts) {
            ONEDAL_ASSERT(count == counts.front());
        }
    }

    std::int64_t minimal_recv_allocation_count = send_ary.get_count();
    allreduce(comm, minimal_recv_allocation_count);
    ONEDAL_ASSERT(recv_ary.get_count() >= minimal_recv_allocation_count);
#endif

    spmd_request request;

    // TODO: Check if `send_ary` and `recv_ary` are in the same context
    __ONEDAL_IF_QUEUE__(send_ary.get_queue(), {
        auto q = send_ary.get_queue().value();
        request = comm.allgather(q,
                                 reinterpret_cast<const byte_t*>(send_ary.get_data()),
                                 send_ary.get_size(),
                                 reinterpret_cast<byte_t*>(recv_ary.get_mutable_data()),
                                 send_ary.get_size());
    });

    __ONEDAL_IF_NO_QUEUE__(send_ary.get_queue(), {
        request = comm.allgather(reinterpret_cast<const byte_t*>(send_ary.get_data()),
                                 send_ary.get_size(),
                                 reinterpret_cast<byte_t*>(recv_ary.get_mutable_data()),
                                 send_ary.get_size());
    });

    return request;
}

} // namespace v1

using v1::is_root_rank;
using v1::if_root_rank;
using v1::if_root_rank_else;
using v1::bcast;
using v1::gather;
using v1::allreduce;
using v1::allgather;

} // namespace oneapi::dal::detail
