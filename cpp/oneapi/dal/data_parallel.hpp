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

#include "oneapi/dal/execution_context.hpp"

namespace oneapi::dal {

namespace detail {
class data_parallel_execution_context;
} // namespace detail

class data_parallel_execution_context : public base {
  public:
    using tag_t = detail::execution_context_tag;
    data_parallel_execution_context() = default;

  private:
    dal::detail::pimpl<detail::data_parallel_execution_context> impl_;
};

template <typename Queue>
inline auto make_context(const Queue& queue) {
    return data_parallel_execution_context();
}

} // namespace oneapi::dal
