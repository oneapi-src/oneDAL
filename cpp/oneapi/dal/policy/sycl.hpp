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

#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::preview {

#ifdef ONEDAL_DATA_PARALLEL
class sycl_policy : public base {
public:
    sycl_policy(const sycl::queue& queue) : internal_policy_(queue) {}

    const sycl::queue& get_queue() const {
        return internal_policy_.get_queue();
    }

private:
    const dal::detail::data_parallel_policy& get_internal_policy() const {
        return internal_policy_;
    }

    dal::detail::data_parallel_policy internal_policy_;
};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_execution_policy<sycl_policy> : std::bool_constant<true> {};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_sycl_policy<sycl_policy> : std::bool_constant<true> {};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_local_policy<sycl_policy> : std::bool_constant<true> {};
#endif

} // namespace oneapi::dal::preview
