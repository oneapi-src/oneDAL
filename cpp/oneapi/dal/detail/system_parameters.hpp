/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <string>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/system_parameters_impl.hpp"

namespace oneapi::dal {

namespace detail {

class system_parameters : public base {
public:
    explicit system_parameters();

    /// C++ related parameters
    cpu_extension get_top_enabled_cpu_extension() const;
    std::uint32_t get_max_number_of_threads() const;

#ifdef ONEDAL_DATA_PARALLEL
    /// DPC++ related parameters
    std::uint32_t get_max_workgroup_size(sycl::queue& queue) const;
#endif

    /// Logging
    std::string dump() const;

#ifdef ONEDAL_DATA_PARALLEL
    std::string dump(sycl::queue& queue) const;
#endif

private:
    detail::pimpl<system_parameters_impl> impl_;
};

} // namespace detail
} // namespace oneapi::dal