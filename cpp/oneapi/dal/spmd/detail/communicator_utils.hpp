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

#include "oneapi/dal/spmd/communicator.hpp"
#include <vector>

namespace ps = oneapi::dal::preview::spmd;

namespace oneapi::dal::detail {

template <typename T>
static constexpr bool is_primitive_v = std::is_arithmetic_v<T>;
template <typename T>
using enable_if_primitive_t = std::enable_if_t<is_primitive_v<T>>;

template <typename memory_access_kind, typename IfBody>
auto if_root_rank(const ps::communicator<memory_access_kind>& comm,
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

template <typename memory_access_kind, typename T, enable_if_primitive_t<T>* = nullptr>
ps::request bcast(const ps::communicator<memory_access_kind>& comm,
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

    ps::request request;
    if (comm.is_root_rank(root)) {
        // `const_cast` is safe here, `bcast` called on the
        // root rank does not modify the values
        if constexpr (!std::is_same_v<memory_access_kind, ps::device_memory_access::none>) {
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

        if constexpr (!std::is_same_v<memory_access_kind, ps::device_memory_access::none>) {
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
template <typename memory_access_kind, typename T, enable_if_primitive_t<T>* = nullptr>
ps::request allreduce(const ps::communicator<memory_access_kind>& comm,
                      const array<T>& ary,
                      const ps::reduce_op& op = ps::reduce_op::sum) {
    if (ary.get_count() == 0) {
        return ps::request{};
    }

    ONEDAL_ASSERT(ary.get_count() > 0);
    ONEDAL_ASSERT(ary.has_mutable_data());

    ps::request request;
    if constexpr (!std::is_same_v<memory_access_kind, ps::device_memory_access::none>) {
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

template <typename memory_access_kind, typename T, enable_if_primitive_t<T>* = nullptr>
ps::request allgather(const ps::communicator<memory_access_kind>& comm,
                      const array<T>& send,
                      const array<T>& recv) {
    if (send.get_count() == 0) {
        ONEDAL_ASSERT(recv.get_count() == 0);
        return ps::request{};
    }

    ONEDAL_ASSERT(send.get_count() > 0);
    ONEDAL_ASSERT(recv.has_mutable_data());

    ps::request request;

    std::vector<std::int64_t> recv_counts(comm.get_rank_count(), send.get_count());
    std::vector<std::int64_t> displs(comm.get_rank_count(), 0);
    std::int64_t total_count = 0;
    for (std::int64_t i = 0; i < send.get_count(); i++) {
        displs[i] = total_count;
        total_count += send.get_count();
    }

    if constexpr (!std::is_same_v<memory_access_kind, ps::device_memory_access::none>) {
        __ONEDAL_IF_QUEUE__(send.get_queue(), {
            auto q = send.get_queue().value();

            ONEDAL_ASSERT(recv.get_queue().has_value());
            ONEDAL_ASSERT(recv.get_queue().value().get_context() == q.get_context());

            request = comm.allgatherv(q,
                                      send.get_data(),
                                      send.get_count(),
                                      recv.get_mutable_data(),
                                      recv_counts.data(),
                                      displs.data());
        });
    }

    __ONEDAL_IF_NO_QUEUE__(send.get_queue(), {
        request = comm.allgatherv(send.get_data(),
                                  send.get_count(),
                                  recv.get_mutable_data(),
                                  recv_counts.data(),
                                  displs.data());
    });

    return request;
}

template <typename memory_access_kind, typename T, enable_if_primitive_t<T>* = nullptr>
ps::request allgather(const ps::communicator<memory_access_kind>& comm,
                      T& scalar,
                      const array<T>& recv) {
    auto send = array<T>::full(1, T(scalar));
    return allgather(comm, send, recv);
}

} // namespace oneapi::dal::detail
