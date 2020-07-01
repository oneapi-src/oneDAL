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

#include "oneapi/dal/algo/kmeans/infer_types.hpp"

namespace oneapi::dal::kmeans::detail {

template <typename Context, typename... Options>
struct ONEAPI_DAL_EXPORT infer_ops_dispatcher {
    infer_result operator()(const Context&, const descriptor_base&, const infer_input&) const;
};

template <typename Descriptor>
struct infer_ops {
    using float_t           = typename Descriptor::float_t;
    using method_t          = method::by_default;
    using input_t           = infer_input;
    using result_t          = infer_result;
    using descriptor_base_t = descriptor_base;

    void validate(const Descriptor& params, const infer_input& input) const {}

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const infer_input& input) const {
        validate(desc, input);
        return infer_ops_dispatcher<Context, float_t, method_t>()(ctx, desc, input);
    }
};

} // namespace oneapi::dal::kmeans::detail
