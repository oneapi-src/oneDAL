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

#include "oneapi/dal/io/csv_data_source/backend/cpu/read_kernel.hpp"
#include "oneapi/dal/io/csv_data_source/backend/gpu/read_kernel.hpp"
#include "oneapi/dal/io/csv_data_source/detail/read_ops.hpp"
#include "oneapi/dal/backend/dispatcher_dpc.hpp"

namespace oneapi::dal::csv_data_source::detail {
using oneapi::dal::detail::data_parallel_policy;

template<>
table read_ops_dispatcher<table, data_parallel_policy>::operator()(const data_parallel_policy& ctx,
                                                                   const params_base& params,
                                                                   const read_input<table>& input) const {
    using kernel_dispatcher_t =
        dal::backend::kernel_dispatcher<backend::read_kernel_cpu<table>,
                                        backend::read_kernel_gpu<table>>;
    return kernel_dispatcher_t{}(ctx, params, input);
}

template struct ONEAPI_DAL_EXPORT read_ops_dispatcher<table, data_parallel_policy>;

} // namespace oneapi::dal::csv_data_source::detail
