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

#include "oneapi/dal/io/csv/detail/read_ops.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/io/csv/backend/cpu/read_kernel.hpp"

namespace oneapi::dal::csv::detail {
using oneapi::dal::detail::host_policy;

template <>
table read_ops_dispatcher<table, host_policy>::operator()(const host_policy& ctx,
                                                          const data_source& ds,
                                                          const read_args<table>& args) const {
    using kernel_dispatcher_t = dal::backend::kernel_dispatcher<backend::read_kernel_cpu<table>>;
    return kernel_dispatcher_t()(ctx, ds, args);
}

template struct ONEAPI_DAL_EXPORT read_ops_dispatcher<table, host_policy>;

} // namespace oneapi::dal::csv::detail
