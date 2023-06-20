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

#include "oneapi/dal/spmd/communicator.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::detail {

template <typename T>
static constexpr bool is_primitive_v = std::is_arithmetic_v<T>;
template <typename T>
using enable_if_primitive_t = std::enable_if_t<is_primitive_v<T>>;

template <typename MemoryAccessKind, typename IfBody>
auto if_root_rank(const spmd::communicator<MemoryAccessKind>& comm,
                  IfBody&& if_body,
                  std::int64_t root = -1) -> decltype(if_body()) {
    if (comm.is_root_rank(root)) {
        return if_body();
    }
    else {
        using return_t = decltype(if_body());
        return return_t{};
    }
}

template <typename MemoryAccessKind, typename T, enable_if_primitive_t<T>* = nullptr>
spmd::request bcast(const spmd::communicator<MemoryAccessKind>& comm,
                    const array<T>& ary,
                    std::int64_t root = -1) {
    std::int64_t count = if_root_rank(
        comm,
        [&]() {
            return ary.get_count();
        },
        root);

    comm.bcast(count, root).wait();
    ONEDAL_ASSERT(ary.get_count() >= count);

    spmd::request request;
    if (comm.is_root_rank(root)) {
        // `const_cast` is safe here, `bcast` called on the
        // root rank does not modify the values
        if constexpr (!std::is_same_v<MemoryAccessKind, spmd::device_memory_access::none>) {
            __ONEDAL_IF_QUEUE__(ary.get_queue(), {
                auto q = ary.get_queue().value();
                request = comm.bcast(q, const_cast<T*>(ary.get_data()), count, {}, root);
            });
        }
        __ONEDAL_IF_NO_QUEUE__(ary.get_queue(), { //
            request = comm.bcast(const_cast<T*>(ary.get_data()), count, root);
        });
    }
    else {
        ONEDAL_ASSERT(ary.has_mutable_data());

        if constexpr (!std::is_same_v<MemoryAccessKind, spmd::device_memory_access::none>) {
            __ONEDAL_IF_QUEUE__(ary.get_queue(), {
                auto q = ary.get_queue().value();
                request = comm.bcast(q, ary.get_mutable_data(), count, {}, root);
            });
        }
        __ONEDAL_IF_NO_QUEUE__(ary.get_queue(), { //
            request = comm.bcast(ary.get_mutable_data(), count, root);
        });
    }

    return request;
}
template <typename MemoryAccessKind, typename T, enable_if_primitive_t<T>* = nullptr>
spmd::request allreduce(const spmd::communicator<MemoryAccessKind>& comm,
                        const array<T>& ary,
                        const spmd::reduce_op& op = spmd::reduce_op::sum) {
    if (ary.get_count() == 0) {
        return spmd::request{};
    }

    ONEDAL_ASSERT(ary.get_count() > 0);
    ONEDAL_ASSERT(ary.has_mutable_data());

    spmd::request request;
    if constexpr (!std::is_same_v<MemoryAccessKind, spmd::device_memory_access::none>) {
        __ONEDAL_IF_QUEUE__(ary.get_queue(), {
            auto q = ary.get_queue().value();
            request =
                comm.allreduce(q, ary.get_data(), ary.get_mutable_data(), ary.get_count(), op, {});
        });
    }

    __ONEDAL_IF_NO_QUEUE__(ary.get_queue(), {
        request = comm.allreduce(ary.get_data(), ary.get_mutable_data(), ary.get_count(), op);
    });

    return request;
}

template <typename MemoryAccessKind, typename T, enable_if_primitive_t<T>* = nullptr>
spmd::request allgatherv(const spmd::communicator<MemoryAccessKind>& comm,
                         const array<T>& send,
                         const array<T>& recv,
                         const std::int64_t* recv_counts,
                         const std::int64_t* displs) {
    ONEDAL_ASSERT(recv.has_mutable_data());

    spmd::request request;

    if constexpr (!std::is_same_v<MemoryAccessKind, spmd::device_memory_access::none>) {
        __ONEDAL_IF_QUEUE__(send.get_queue(), {
            auto q = send.get_queue().value();
            ONEDAL_ASSERT(recv.get_queue().has_value());
            ONEDAL_ASSERT(recv.get_queue().value().get_context() == q.get_context());

            request = comm.allgatherv(q,
                                      send.get_data(),
                                      send.get_count(),
                                      recv.get_mutable_data(),
                                      recv_counts,
                                      displs);
        });
    }

    __ONEDAL_IF_NO_QUEUE__(send.get_queue(), {
        request = comm.allgatherv(send.get_data(),
                                  send.get_count(),
                                  recv.get_mutable_data(),
                                  recv_counts,
                                  displs);
    });

    return request;
}

template <typename MemoryAccessKind, typename T, enable_if_primitive_t<T>* = nullptr>
spmd::request allgather(const spmd::communicator<MemoryAccessKind>& comm,
                        const array<T>& send,
                        const array<T>& recv) {
    if (send.get_count() == 0) {
        ONEDAL_ASSERT(recv.get_count() == 0);
        return spmd::request{};
    }

    ONEDAL_ASSERT(send.get_count() > 0);
    ONEDAL_ASSERT(recv.has_mutable_data());

    spmd::request request;

    auto recv_counts = array<std::int64_t>::full(comm.get_rank_count(), send.get_count());
    auto displs = array<std::int64_t>::zeros(comm.get_rank_count());
    auto recv_counts_ptr = recv_counts.get_data();
    auto displs_ptr = displs.get_mutable_data();
    std::int64_t total_count = 0;
    for (std::int64_t i = 0; i < comm.get_rank_count(); i++) {
        displs_ptr[i] = total_count;
        total_count += recv_counts_ptr[i];
    }

    return allgatherv(comm, send, recv, recv_counts.get_data(), displs.get_data());
}

template <typename MemoryAccessKind, typename T, enable_if_primitive_t<T>* = nullptr>
spmd::request allgather(const spmd::communicator<MemoryAccessKind>& comm,
                        const T& scalar,
                        const array<T>& recv) {
    auto send = array<T>::full(1, T(scalar));
    return allgather(comm, send, recv);
}

template <typename MemoryAccessKind, typename T, enable_if_primitive_t<T>* = nullptr>
spmd::request sendrecv_replace(const spmd::communicator<MemoryAccessKind>& comm,
                               const array<T>& buf,
                               std::int64_t destination_rank,
                               std::int64_t source_rank) {
    ONEDAL_ASSERT(buf.has_mutable_data());
    ONEDAL_ASSERT(destination_rank >= 0);
    ONEDAL_ASSERT(source_rank >= 0);

    spmd::request request;

    if constexpr (!std::is_same_v<MemoryAccessKind, spmd::device_memory_access::none>) {
        __ONEDAL_IF_QUEUE__(buf.get_queue(), {
            ONEDAL_ASSERT(buf.get_queue().has_value());
            auto q = buf.get_queue().value();
            request = comm.sendrecv_replace(q,
                                            buf.get_mutable_data(),
                                            buf.get_count(),
                                            destination_rank,
                                            source_rank);
        });
    }

    __ONEDAL_IF_NO_QUEUE__(buf.get_queue(), {
        request = comm.sendrecv_replace(buf.get_mutable_data(),
                                        buf.get_count(),
                                        destination_rank,
                                        source_rank);
    });

    return request;
}

} // namespace oneapi::dal::detail
