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

#include "oneapi/dal/io/csv/read_types.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::csv::detail {

template <typename table, typename Context, typename... Options>
struct ONEAPI_DAL_EXPORT read_ops_dispatcher {
    table operator()(const Context&, const data_source_base&, const read_args<table>&) const;
};

template <typename table, typename Descriptor>
struct read_ops {
    using input_t           = read_args<table>;
    using result_t          = table;
    using descriptor_base_t = data_source_base;

    void check_preconditions(const Descriptor& data_source, const input_t& input) const {
    }

    void check_postconditions(const Descriptor& data_source,
                              const input_t& input,
                              const result_t& result) const {
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const read_args<table>& args) const {
        check_preconditions(desc, args);
        const auto result = read_ops_dispatcher<table, Context>()(ctx, desc, args);
        check_postconditions(desc, args, result);
        return result;
    }
};

} // namespace oneapi::dal::csv::detail
