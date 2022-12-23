/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <type_traits>

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <typename Param>
class logloss_test : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
    using float_t = std::tuple_element_t<0, Param>;
};

using logloss_types = COMBINE_TYPES((float, double));

} // namespace oneapi::dal::backend::primitives::test
