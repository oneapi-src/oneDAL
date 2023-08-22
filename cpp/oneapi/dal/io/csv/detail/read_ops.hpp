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
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/io/csv/read_types.hpp"
#include "oneapi/dal/io/csv/detail/select_kernel.hpp"

namespace oneapi::dal::csv::detail {

template <typename Object, typename Policy, typename... Options>
struct read_ops_dispatcher;

template <typename Object>
struct read_ops_dispatcher<Object, dal::detail::host_policy> {
    Object operator()(const dal::detail::host_policy& policy,
                      const data_source_base& ds,
                      const dal::preview::csv::read_args<Object>& args) const {
        static auto impl =
            dal::preview::csv::detail::get_backend<dal::detail::host_policy,
                                                   data_source_base,
                                                   dal::preview::csv::read_args<Object>>(ds, args);
        return (*impl)(policy, ds, args);
    }
};

template <>
struct ONEDAL_EXPORT read_ops_dispatcher<table, dal::detail::host_policy> {
    table operator()(const dal::detail::host_policy& policy,
                     const data_source_base& ds,
                     const dal::csv::read_args<table>& args) const;
};

#ifdef ONEDAL_DATA_PARALLEL

template <>
struct ONEDAL_EXPORT read_ops_dispatcher<table, dal::detail::data_parallel_policy> {
    table operator()(const dal::detail::data_parallel_policy& ctx,
                     const data_source_base& ds,
                     const dal::csv::read_args<table>& args) const;
};

#endif

template <typename Object, typename DataSource>
struct read_ops;

template <typename Object>
struct read_ops<Object, data_source> {
    using input_t = dal::preview::csv::read_args<Object>;
    using result_t = Object;

    void check_preconditions(const data_source_base& ds, const input_t& args) const {}

    void check_postconditions(const data_source_base& ds,
                              const input_t& args,
                              const result_t& result) const {}

    template <typename Policy>
    auto operator()(const Policy& ctx, const data_source_base& ds, const input_t& args) const {
        check_preconditions(ds, args);
        auto result = read_ops_dispatcher<Object, Policy>()(ctx, ds, args);
        check_postconditions(ds, args, result);
        return result;
    }
};

template <>
struct read_ops<table, data_source> {
    using input_t = read_args<table>;
    using result_t = table;

    void check_preconditions(const data_source_base& ds, const input_t& args) const {}

    void check_postconditions(const data_source_base& ds,
                              const input_t& args,
                              const result_t& result) const {}

    template <typename Policy>
    auto operator()(const Policy& ctx, const data_source_base& ds, const input_t& args) const {
        check_preconditions(ds, args);
        const auto result = read_ops_dispatcher<table, Policy>()(ctx, ds, args);
        check_postconditions(ds, args, result);
        return result;
    }
};

} // namespace oneapi::dal::csv::detail
