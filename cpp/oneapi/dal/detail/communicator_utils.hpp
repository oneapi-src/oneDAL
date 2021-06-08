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
#include "oneapi/dal/detail/communicator.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/archives.hpp"

namespace oneapi::dal::detail {
namespace v1 {

inline bool is_root_rank(const dal::detail::spmd_communicator& comm) {
    return comm.get_rank() == comm.get_root_rank();
}

template <typename IfBody>
inline auto if_root_rank(const dal::detail::spmd_communicator& comm, IfBody&& if_body) {
    if (is_root_rank(comm)) {
        return if_body();
    }
    else {
        using return_t = decltype(if_body());
        return return_t{};
    }
}

template <typename IfBody, typename ElseBody>
inline auto if_root_rank_else(const dal::detail::spmd_communicator& comm,
                              IfBody&& if_body,
                              ElseBody&& else_body) {
    if (is_root_rank(comm)) {
        return if_body();
    }
    else {
        return else_body();
    }
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline void bcast(const dal::detail::spmd_communicator& comm,
                  T* buf,
                  std::int64_t count,
                  std::int64_t root) {
    const std::int64_t count_in_bytes =
        dal::detail::check_mul_overflow<std::int64_t>(count, sizeof(T));
    auto request = comm.bcast(reinterpret_cast<byte_t*>(buf), count_in_bytes, root);
    request.wait();
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline void bcast(const dal::detail::spmd_communicator& comm, T* buf, std::int64_t count) {
    bcast(comm, buf, count, comm.get_root_rank());
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline void bcast(const dal::detail::spmd_communicator& comm, T& value) {
    bcast(comm, &value, 1, comm.get_root_rank());
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline void bcast(const dal::detail::spmd_communicator& comm, array<T>& ary) {
    std::int64_t count = if_root_rank(comm, [&]() {
        return ary.get_count();
    });

    bcast(comm, count);

    if (is_root_rank(comm)) {
        // `const_cast` is safe here, `bcast` called on the
        // root rank does not modify the values
        bcast(comm, const_cast<T*>(ary.get_data()), count);
    }
    else {
        if (count > ary.get_count()) {
            ary = array<T>::empty(count);
        }
        else {
            ary.need_mutable_data();
        }

        bcast(comm, ary.get_mutable_data(), count);
    }
}

template <typename T, dal::detail::enable_if_user_serializable_t<T>* = nullptr>
inline void bcast(const dal::detail::spmd_communicator& comm, T& send) {
    array<byte_t> binary;
    byte_t* binary_ptr = nullptr;
    std::int64_t binary_count = 0;

    if (is_root_rank(comm)) {
        dal::detail::binary_output_archive out_archive;
        dal::detail::serialize(send, out_archive);
        binary = out_archive.to_array();

        ONEDAL_ASSERT(binary.has_mutable_data());
        binary_ptr = binary.get_mutable_data();
        binary_count = binary.get_count();
    }

    bcast(comm, binary_count);
    ONEDAL_ASSERT(binary_count > 0);

    if (!is_root_rank(comm)) {
        binary = array<byte_t>::empty(binary_count);
        binary_ptr = binary.get_mutable_data();
    }

    ONEDAL_ASSERT(binary_ptr);
    bcast(comm, binary_ptr, binary_count);

    if (!is_root_rank(comm)) {
        dal::detail::binary_input_archive in_archive{ binary };
        dal::detail::deserialize(send, in_archive);
    }
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline void gather(const dal::detail::spmd_communicator& comm,
                   const T* send_buf,
                   std::int64_t send_count,
                   T* recv_buf,
                   std::int64_t recv_count,
                   std::int64_t root) {
    const std::int64_t send_count_in_bytes =
        dal::detail::check_mul_overflow<std::int64_t>(send_count, sizeof(T));
    const std::int64_t recv_count_in_bytes =
        dal::detail::check_mul_overflow<std::int64_t>(recv_count, sizeof(T));

    auto request = comm.gather(reinterpret_cast<const byte_t*>(send_buf),
                               send_count_in_bytes,
                               reinterpret_cast<byte_t*>(recv_buf),
                               recv_count_in_bytes,
                               root);
    request.wait();
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline void gather(const dal::detail::spmd_communicator& comm,
                   const T* send_buf,
                   std::int64_t send_count,
                   T* recv_buf,
                   std::int64_t recv_count) {
    gather(comm, send_buf, send_count, recv_buf, recv_count, comm.get_root_rank());
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline void gather(const dal::detail::spmd_communicator& comm,
                   const T* send_buf,
                   std::int64_t send_count,
                   T* recv_buf) {
    gather(comm, send_buf, send_count, recv_buf, send_count, comm.get_root_rank());
}

template <typename T, dal::detail::enable_if_trivially_serializable_t<T>* = nullptr>
inline std::vector<T> gather(const dal::detail::spmd_communicator& comm, const T& send) {
    std::vector<T> recv;
    if (is_root_rank(comm)) {
        recv.resize(comm.get_rank_count());
    }
    gather(comm, &send, 1, recv.data());
    return recv;
}

template <typename T, dal::detail::enable_if_user_serializable_t<T>* = nullptr>
inline std::vector<T> gather(const dal::detail::spmd_communicator& comm, const T& send) {
    dal::detail::binary_output_archive out_archive;
    dal::detail::serialize(send, out_archive);

    const auto send_binary = out_archive.to_array();
    const std::vector<std::int64_t> send_buffer_sizes = gather(comm, send_binary.get_count());

    const std::vector<std::int64_t> root_displs = if_root_rank(comm, [&]() {
        ONEDAL_ASSERT(send_buffer_sizes.size() > 0);
        std::vector<std::int64_t> displs(comm.get_rank_count(), std::int64_t(0));
        for (std::int64_t i = 1; i < comm.get_rank_count(); i++) {
            displs[i] = displs[i - 1] + send_buffer_sizes[i - 1];
        }
        return displs;
    });

    const array<byte_t> root_buffer = if_root_rank(comm, [&]() {
        ONEDAL_ASSERT(send_buffer_sizes.size() > 0);
        const std::int64_t root_buffer_size =
            std::accumulate(send_buffer_sizes.begin(), send_buffer_sizes.end(), std::int64_t(0));
        return array<byte_t>::empty(root_buffer_size);
    });

    byte_t* root_buffer_ptr = if_root_rank(comm, [&]() {
        ONEDAL_ASSERT(root_buffer.get_count() > 0);
        return root_buffer.get_mutable_data();
    });

    auto request = comm.gatherv(send_binary.get_data(),
                                send_binary.get_count(),
                                root_buffer_ptr,
                                send_buffer_sizes.data(),
                                root_displs.data(),
                                comm.get_root_rank());
    request.wait();

    std::vector<T> gathered;
    if (is_root_rank(comm)) {
        ONEDAL_ASSERT(root_buffer_ptr);
        ONEDAL_ASSERT(root_displs.size() > 0);
        ONEDAL_ASSERT(send_buffer_sizes.size() > 0);

        gathered.reserve(comm.get_rank_count());

        for (std::int64_t i = 0; i < comm.get_rank_count(); i++) {
            const byte_t* root_buffer_displs_ptr = root_buffer_ptr + root_displs[i];
            const std::int64_t root_buffer_displs_size = send_buffer_sizes[i];

            dal::detail::binary_input_archive in_archive{ root_buffer_displs_ptr,
                                                          root_buffer_displs_size };

            T deserialized;
            dal::detail::deserialize(deserialized, in_archive);

            gathered.push_back(std::move(deserialized));
        }
    }

    return gathered;
}

} // namespace v1

using v1::is_root_rank;
using v1::if_root_rank;
using v1::if_root_rank_else;
using v1::bcast;
using v1::gather;

} // namespace oneapi::dal::detail
