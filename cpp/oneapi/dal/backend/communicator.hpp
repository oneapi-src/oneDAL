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
#include "oneapi/dal/detail/communicator_utils.hpp"

namespace oneapi::dal::backend {

class spmd_communicator {
public:
    spmd_communicator() = default;
    explicit spmd_communicator(const dal::detail::spmd_communicator& comm) : comm_(comm) {}

    std::int64_t get_rank() const {
        return comm_.get_rank();
    }

    std::int64_t get_rank_count() const {
        return comm_.get_rank_count();
    }

    std::int64_t get_root_rank() const {
        return comm_.get_root_rank();
    }

    bool is_root() const {
        return dal::detail::is_root_rank(comm_);
    }

    template <typename IfBody>
    auto if_root(IfBody&& if_body) const {
        return dal::detail::if_root_rank(comm_, std::forward<IfBody>(if_body));
    }

    template <typename IfBody, typename ElseBody>
    auto if_root_else(IfBody&& if_body, ElseBody&& else_body) const {
        return dal::detail::if_root_rank_else(comm_,
                                              std::forward<IfBody>(if_body),
                                              std::forward<ElseBody>(else_body));
    }

    template <typename T>
    void bcast(T* buf, std::int64_t count, std::int64_t root) const {
        dal::detail::bcast(comm_, buf, count, root);
    }

    template <typename T>
    void bcast(T* buf, std::int64_t count) const {
        dal::detail::bcast(comm_, buf, count);
    }

    template <typename T>
    void bcast(T& value) const {
        dal::detail::bcast(comm_, value);
    }

    template <typename T>
    void gather(const T* send_buf,
                std::int64_t send_count,
                T* recv_buf,
                std::int64_t recv_count,
                std::int64_t root) const {
        dal::detail::gather(comm_, send_buf, send_count, recv_buf, recv_count, root);
    }

    template <typename T>
    void gather(const T* send_buf,
                std::int64_t send_count,
                T* recv_buf,
                std::int64_t recv_count) const {
        dal::detail::gather(comm_, send_buf, send_count, recv_buf, recv_count);
    }

    template <typename T>
    void gather(const T* send_buf, std::int64_t send_count, T* recv_buf) const {
        dal::detail::gather(comm_, send_buf, send_count, recv_buf);
    }

    template <typename T>
    std::vector<T> gather(const T& send) const {
        return dal::detail::gather(comm_, send);
    }

private:
    dal::detail::spmd_communicator comm_;
};

} // namespace oneapi::dal::backend
