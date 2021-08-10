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

#include <tuple>
#include <memory>
#include <iostream>
#include <type_traits>

#include <fmt/core.h>

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/test/engine/catch.hpp"
#include "oneapi/dal/test/engine/common_base.hpp"
#include "oneapi/dal/test/engine/macro.hpp"
#include "oneapi/dal/test/engine/type_traits.hpp"

// Workaround DPC++ Compiler's warning on unused
// variable declared by Catch2's TEST_CASE macro
#ifdef __clang__
#define _TE_DISABLE_UNUSED_VARIABLE \
    _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wunused-variable\"")

#define _TE_ENABLE_UNUSED_VARIABLE _Pragma("clang diagnostic pop")

#undef TEST_CASE
#define TEST_CASE(...)                   \
    _TE_DISABLE_UNUSED_VARIABLE          \
    INTERNAL_CATCH_TESTCASE(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE
#define TEMPLATE_TEST_CASE(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                    \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_LIST_TEST_CASE
#define TEMPLATE_LIST_TEST_CASE(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                         \
    INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE_SIG
#define TEMPLATE_TEST_CASE_SIG(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                        \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEST_CASE_METHOD
#define TEST_CASE_METHOD(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                  \
    INTERNAL_CATCH_TEST_CASE_METHOD(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE_METHOD
#define TEMPLATE_TEST_CASE_METHOD(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                           \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_LIST_TEST_CASE_METHOD
#define TEMPLATE_LIST_TEST_CASE_METHOD(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                                \
    INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE_METHOD_SIG
#define TEMPLATE_TEST_CASE_METHOD_SIG(...)                    \
    _TE_DISABLE_UNUSED_VARIABLE                               \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG(__VA_ARGS__) \
    _TE_ENABLE_UNUSED_VARIABLE
#endif // __clang__

// Shortcuts for Catch2 defines
#define TEST                 TEST_CASE
#define TEMPLATE_TEST        TEMPLATE_TEST_CASE
#define TEMPLATE_LIST_TEST   TEMPLATE_LIST_TEST_CASE
#define TEMPLATE_SIG_TEST    TEMPLATE_TEST_CASE_SIG
#define TEST_M               TEST_CASE_METHOD
#define TEMPLATE_TEST_M      TEMPLATE_TEST_CASE_METHOD
#define TEMPLATE_LIST_TEST_M TEMPLATE_LIST_TEST_CASE_METHOD
#define TEMPLATE_SIG_TEST_M  TEMPLATE_TEST_CASE_METHOD_SIG

#define SKIP_IF(condition) \
    if (condition) {       \
        SUCCEED();         \
        return;            \
    }