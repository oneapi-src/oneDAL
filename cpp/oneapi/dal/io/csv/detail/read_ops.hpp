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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/io/csv/read_types.hpp"

namespace oneapi::dal::csv::detail {
namespace v1 {

template <typename Object, typename Policy, typename... Options>
struct read_ops_dispatcher {
    Object operator()(const Policy&, const data_source_base&, const read_args<Object>&) const;
};

template <typename Object, typename DataSource>
struct read_ops;

template <typename Object>
struct read_ops<Object, data_source> {
    static_assert(std::is_same_v<Object, table>, "CSV data source is defined only for table");

    using args_t = read_args<Object>;
    using result_t = Object;

    void check_preconditions(const data_source_base& ds, const args_t& args) const {}

    void check_postconditions(const data_source_base& ds,
                              const args_t& args,
                              const result_t& result) const {}

    template <typename Policy>
    auto operator()(const Policy& ctx, const data_source_base& ds, const args_t& args) const {
        check_preconditions(ds, args);
        const auto result = read_ops_dispatcher<Object, Policy>()(ctx, ds, args);
        check_postconditions(ds, args, result);
        return result;
    }
};

} // namespace v1

using v1::read_ops;

} // namespace oneapi::dal::csv::detail
