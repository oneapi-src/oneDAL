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

#include "oneapi/dal/detail/spmd_policy.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test::engine {

template <typename... Args>
inline auto spmd_train(host_test_policy& policy,
                       const spmd::communicator<spmd::device_memory_access::none>& comm,
                       Args&&... args) {
    return dal::train(
        dal::detail::spmd_policy<dal::detail::host_policy>{ dal::detail::host_policy{}, comm },
        std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
inline auto spmd_train(device_test_policy& policy,
                       const spmd::communicator<spmd::device_memory_access::usm>& comm,
                       Args&&... args) {
    dal::detail::data_parallel_policy local_policy{ policy.get_queue() };
    dal::detail::spmd_policy<detail::data_parallel_policy> spmd_policy{ local_policy, comm };
    return dal::train(spmd_policy, std::forward<Args>(args)...);
}
#endif

template <typename... Args>
inline auto spmd_finalize_train(host_test_policy& policy,
                                const spmd::communicator<spmd::device_memory_access::none>& comm,
                                Args&&... args) {
    return dal::finalize_train(dal::detail::spmd_policy{ dal::detail::host_policy{}, comm },
                               std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
inline auto spmd_finalize_train(device_test_policy& policy,
                                const spmd::communicator<spmd::device_memory_access::usm>& comm,
                                Args&&... args) {
    dal::detail::data_parallel_policy local_policy{ policy.get_queue() };
    dal::detail::spmd_policy<detail::data_parallel_policy> spmd_policy{ local_policy, comm };
    return dal::finalize_train(spmd_policy, std::forward<Args>(args)...);
}
#endif

template <typename... Args>
inline auto spmd_infer(host_test_policy& policy,
                       const spmd::communicator<spmd::device_memory_access::none>& comm,
                       Args&&... args) {
    return dal::infer(dal::detail::spmd_policy{ dal::detail::host_policy{}, comm },
                      std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
inline auto spmd_infer(device_test_policy& policy,
                       const spmd::communicator<spmd::device_memory_access::usm>& comm,
                       Args&&... args) {
    dal::detail::data_parallel_policy local_policy{ policy.get_queue() };
    dal::detail::spmd_policy<detail::data_parallel_policy> spmd_policy{ local_policy, comm };
    return dal::infer(spmd_policy, std::forward<Args>(args)...);
}
#endif

template <typename... Args>
inline auto spmd_compute(host_test_policy& policy,
                         const spmd::communicator<spmd::device_memory_access::none>& comm,
                         Args&&... args) {
    return dal::compute(dal::detail::spmd_policy{ dal::detail::host_policy{}, comm },
                        std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
inline auto spmd_compute(device_test_policy& policy,
                         const spmd::communicator<spmd::device_memory_access::usm>& comm,
                         Args&&... args) {
    dal::detail::data_parallel_policy local_policy{ policy.get_queue() };
    dal::detail::spmd_policy<detail::data_parallel_policy> spmd_policy{ local_policy, comm };
    return dal::compute(spmd_policy, std::forward<Args>(args)...);
}
#endif

template <typename... Args>
inline auto spmd_finalize_compute(host_test_policy& policy,
                                  const spmd::communicator<spmd::device_memory_access::none>& comm,
                                  Args&&... args) {
    return dal::finalize_compute(dal::detail::spmd_policy{ dal::detail::host_policy{}, comm },
                                 std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
inline auto spmd_finalize_compute(device_test_policy& policy,
                                  const spmd::communicator<spmd::device_memory_access::usm>& comm,
                                  Args&&... args) {
    dal::detail::data_parallel_policy local_policy{ policy.get_queue() };
    dal::detail::spmd_policy<detail::data_parallel_policy> spmd_policy{ local_policy, comm };
    return dal::finalize_compute(spmd_policy, std::forward<Args>(args)...);
}
#endif

} // namespace oneapi::dal::test::engine
