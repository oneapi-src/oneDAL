/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/detail/common.hpp"

#ifndef ONEDAL_ENABLE_ASSERT
#define ONEDAL_ASSERT_SUM_OVERFLOW(...)
#define ONEDAL_ASSERT_MUL_OVERFLOW(...)
#else
#define ONEDAL_ASSERT_SUM_OVERFLOW(Data, first, second)                                       \
    {                                                                                         \
        static_assert(std::is_integral_v<Data>, "The check requires integral operands");      \
        Data result;                                                                          \
        ONEDAL_ASSERT(oneapi::dal::detail::integer_overflow_ops<Data>{}.is_safe_sum((first),  \
                                                                                    (second), \
                                                                                    result),  \
                      "Sum overflow assertion failed with operands" #first " and " #second)   \
    }

#define ONEDAL_ASSERT_MUL_OVERFLOW(Data, first, second)                                       \
    {                                                                                         \
        static_assert(std::is_integral_v<Data>, "The check requires integral operands");      \
        Data result;                                                                          \
        ONEDAL_ASSERT(oneapi::dal::detail::integer_overflow_ops<Data>{}.is_safe_mul((first),  \
                                                                                    (second), \
                                                                                    result),  \
                      "Mul overflow assertion failed with operands" #first " and " #second)   \
    }
#endif
