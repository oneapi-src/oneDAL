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

#pragma once

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/communicator.hpp"

namespace oneapi::dal::backend {

/// Implementation of SPMD communicator for one-rank system
class fake_spmd_communicator : public dal::detail::spmd_communicator {
public:
    fake_spmd_communicator();
};

/// Wrapper over public SPMD request.
/// The event-like object seems more natural to data-parallel algorithms
/// which are built on top of SYCL events.
class communicator_event {
public:
    communicator_event() = default;
    communicator_event(const dal::detail::spmd_request& req) : public_req_(req) {}
    communicator_event(dal::detail::spmd_request&& req) : public_req_(std::move(req)) {}

    void wait() {
        public_req_.wait();
    }

private:
    dal::detail::spmd_request public_req_;
};

/// Wrapper over public SPMD communicator.
/// The additional layer of abstraction is added to have more flexibility in changing the
/// communicator interface which algorithms depend on. For example, this can be used to add
/// collective operation overloading for internal classes such as `ndarray`.
class communicator {
public:
    /// Creates communicator for one-rank system
    communicator() : public_comm_(fake_spmd_communicator{}), is_distributed_(false) {}

    /// Creates communicator based on public SPMD interface
    communicator(const dal::detail::spmd_communicator& comm)
            : public_comm_(comm),
              is_distributed_(true) {}

    bool is_distributed() const {
        return is_distributed_;
    }

    void barrier() const {
        public_comm_.barrier();
    }

    template <typename... Args>
    communicator_event bcast(Args&&... args) const {
        return public_comm_.bcast(std::forward<Args>(args)...);
    }

    template <typename... Args>
    communicator_event gather(Args&&... args) const {
        return public_comm_.gather(std::forward<Args>(args)...);
    }

    template <typename... Args>
    communicator_event gatherv(Args&&... args) const {
        return public_comm_.gatherv(std::forward<Args>(args)...);
    }

    template <typename... Args>
    communicator_event allgather(Args&&... args) const {
        return public_comm_.allgather(std::forward<Args>(args)...);
    }

    template <typename... Args>
    communicator_event allreduce(Args&&... args) const {
        return public_comm_.allreduce(std::forward<Args>(args)...);
    }

private:
    dal::detail::spmd_communicator public_comm_;
    bool is_distributed_;
};

} // namespace oneapi::dal::backend
