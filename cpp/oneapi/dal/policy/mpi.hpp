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

#include "oneapi/dal/policy/host.hpp"
#include "oneapi/dal/policy/sycl.hpp"
#include "oneapi/dal/detail/distributed/mpi_communicator.hpp"

namespace oneapi::dal::preview {

template <typename LocalPolicy>
class mpi_policy : public base {
    static_assert(is_execution_policy_v<LocalPolicy>,
                  "LocalPolicy must be an execution policy type");

public:
    template <typename T = LocalPolicy, std::enable_if_t<is_host_policy<LocalPolicy>>* = nullptr>
    mpi_policy(const LocalPolicy& local_policy, const MPI_Comm& mpi_comm = MPI_COMM_WORLD)
        local_policy_(local_policy),
        comm_(detail::mpi_communicator{ mpi_comm }) {}

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T = LocalPolicy, std::enable_if_t<is_sycl_policy<LocalPolicy>>* = nullptr>
    mpi_policy(const LocalPolicy& local_policy, const MPI_Comm& mpi_comm = MPI_COMM_WORLD)
        local_policy_(local_policy),
        comm_(detail::mpi_communicator{ mpi_comm, local_policy.get_queue() }) {}
#endif

    mpi_policy& set_root_rank(std::int64_t root_rank) {
        comm_.set_root_rank(root_rank);
        return *this;
    }

private:
    LocalPolicy local_policy_;
    dal::detail::mpi_communicator comm_;
};

template <typename LocalPolicy>
struct is_distributed_policy<mpi_policy<LocalPolicy>> : std::bool_constant<true> {};

template <>
struct is_host_policy<mpi_policy<host_policy>> : std::bool_constant<true> {};

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_sycl_policy<mpi_policy<sycl_policy>> : std::bool_constant<true> {};
#endif

} // namespace oneapi::dal::preview
