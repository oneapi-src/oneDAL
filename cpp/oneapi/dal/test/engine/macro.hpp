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

// Disable clang-format as it dramatically
// affects redability of macro definitions
// clang-format off

#define _TE_STRINGIFY(x) #x
#define _TE_EXPAND(...)  __VA_ARGS__
#define _TE_UNPACK(x)    _TE_EXPAND x

#define _TE_CONCAT_2(_1, _2)             _1##_2
#define _TE_CONCAT_3(_1, _2, _3)         _1##_2##_3
#define _TE_CONCAT_4(_1, _2, _3, _4)     _1##_2##_3##_4
#define _TE_CONCAT_5(_1, _2, _3, _4, _5) _1##_2##_3##_4##_5

#define _TE_NARGS_GET(_1, _2, _3, _4, _5, N, ...) N
#define _TE_NARGS(...) _TE_NARGS_GET(__VA_ARGS__, 5, 4, 3, 2, 1, 0)

#define _TE_FOR_EACH_0(ctx, a)
#define _TE_FOR_EACH_1(ctx, a, x)      a(ctx, x) _TE_FOR_EACH_0(ctx, a)
#define _TE_FOR_EACH_2(ctx, a, x, ...) a(ctx, x) _TE_FOR_EACH_1(ctx, a, __VA_ARGS__)
#define _TE_FOR_EACH_3(ctx, a, x, ...) a(ctx, x) _TE_FOR_EACH_2(ctx, a, __VA_ARGS__)
#define _TE_FOR_EACH_4(ctx, a, x, ...) a(ctx, x) _TE_FOR_EACH_3(ctx, a, __VA_ARGS__)
#define _TE_FOR_EACH_5(ctx, a, x, ...) a(ctx, x) _TE_FOR_EACH_4(ctx, a, __VA_ARGS__)
#define _TE_FOR_EACH_6(ctx, a, x, ...) a(ctx, x) _TE_FOR_EACH_5(ctx, a, __VA_ARGS__)
#define _TE_FOR_EACH_7(ctx, a, x, ...) a(ctx, x) _TE_FOR_EACH_6(ctx, a, __VA_ARGS__)
#define _TE_FOR_EACH_8(ctx, a, x, ...) a(ctx, x) _TE_FOR_EACH_7(ctx, a, __VA_ARGS__)
#define _TE_FOR_EACH_9(ctx, a, x, ...) a(ctx, x) _TE_FOR_EACH_8(ctx, a, __VA_ARGS__)

#define _TE_FOR_EACH_(N, ctx, action, ...) \
    _TE_CONCAT_2(_TE_FOR_EACH_, N)(ctx, action, __VA_ARGS__)

#define _TE_FOR_EACH(ctx, action, ...) \
    _TE_FOR_EACH_(_TE_NARGS(__VA_ARGS__), ctx, action, __VA_ARGS__)

#define _TE_GET_N1_0(_0) _0
#define _TE_GET_N1_1(_0) _0
#define _TE_GET_N1_2(_0) _0

#define _TE_GET_N2_0(_0, _1) _0
#define _TE_GET_N2_1(_0, _1) _1
#define _TE_GET_N2_2(_0, _1) _1

#define _TE_GET_N3_0(_0, _1, _2) _0
#define _TE_GET_N3_1(_0, _1, _2) _1
#define _TE_GET_N3_2(_0, _1, _2) _2

#define _TE_GET_N(N, i, ...) _TE_CONCAT_4(_TE_GET_N, N, _, i)(__VA_ARGS__)

#define _TE_GET_N_0(...) _TE_GET_N(_TE_NARGS(__VA_ARGS__), 0, __VA_ARGS__)
#define _TE_GET_N_1(...) _TE_GET_N(_TE_NARGS(__VA_ARGS__), 1, __VA_ARGS__)
#define _TE_GET_N_2(...) _TE_GET_N(_TE_NARGS(__VA_ARGS__), 2, __VA_ARGS__)

#define _TE_GET_0(x) _TE_EXPAND(_TE_GET_N_0 x)
#define _TE_GET_1(x) _TE_EXPAND(_TE_GET_N_1 x)
#define _TE_GET_2(x) _TE_EXPAND(_TE_GET_N_2 x)
