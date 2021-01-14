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

// Disable clang-format as it dramatically
// affects redability of macro definitions
// clang-format off

#define _TS_STRINGIFY(x) #x
#define _TS_EXPAND(...)  __VA_ARGS__
#define _TS_UNPACK(x)    _TS_EXPAND x

#define _TS_CONCAT_2(_1, _2)             _1##_2
#define _TS_CONCAT_3(_1, _2, _3)         _1##_2##_3
#define _TS_CONCAT_4(_1, _2, _3, _4)     _1##_2##_3##_4
#define _TS_CONCAT_5(_1, _2, _3, _4, _5) _1##_2##_3##_4##_5

#define _TS_NARGS_GET(_1, _2, _3, _4, _5, N, ...) N
#define _TS_NARGS(...) _TS_NARGS_GET(__VA_ARGS__, 5, 4, 3, 2, 1, 0)

#define _TS_FOR_EACH_0(ctx, a)
#define _TS_FOR_EACH_1(ctx, a, x)      a(ctx, x) _TS_FOR_EACH_0(ctx, a)
#define _TS_FOR_EACH_2(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_1(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_3(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_2(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_4(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_3(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_5(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_4(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_6(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_5(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_7(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_6(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_8(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_7(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_9(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_8(ctx, a, __VA_ARGS__)

#define _TS_FOR_EACH_(N, ctx, action, ...) \
    _TS_CONCAT_2(_TS_FOR_EACH_, N)(ctx, action, __VA_ARGS__)

#define _TS_FOR_EACH(ctx, action, ...) \
    _TS_FOR_EACH_(_TS_NARGS(__VA_ARGS__), ctx, action, __VA_ARGS__)

#define _TS_GET_N1_0(_0) _0
#define _TS_GET_N1_1(_0) _0
#define _TS_GET_N1_2(_0) _0

#define _TS_GET_N2_0(_0, _1) _0
#define _TS_GET_N2_1(_0, _1) _1
#define _TS_GET_N2_2(_0, _1) _1

#define _TS_GET_N3_0(_0, _1, _2) _0
#define _TS_GET_N3_1(_0, _1, _2) _1
#define _TS_GET_N3_2(_0, _1, _2) _2

#define _TS_GET_N(N, i, ...) _TS_CONCAT_4(_TS_GET_N, N, _, i)(__VA_ARGS__)

#define _TS_GET_N_0(...) _TS_GET_N(_TS_NARGS(__VA_ARGS__), 0, __VA_ARGS__)
#define _TS_GET_N_1(...) _TS_GET_N(_TS_NARGS(__VA_ARGS__), 1, __VA_ARGS__)
#define _TS_GET_N_2(...) _TS_GET_N(_TS_NARGS(__VA_ARGS__), 2, __VA_ARGS__)

#define _TS_GET_0(x) _TS_EXPAND(_TS_GET_N_0 x)
#define _TS_GET_1(x) _TS_EXPAND(_TS_GET_N_1 x)
#define _TS_GET_2(x) _TS_EXPAND(_TS_GET_N_2 x)

#define _TS_PAIR_00(x, y) (_TS_GET_0(x), _TS_GET_0(y))
#define _TS_PAIR_01(x, y) (_TS_GET_0(x), _TS_GET_1(y))
#define _TS_PAIR_02(x, y) (_TS_GET_0(x), _TS_GET_2(y))
#define _TS_PAIR_10(x, y) (_TS_GET_1(x), _TS_GET_0(y))
#define _TS_PAIR_11(x, y) (_TS_GET_1(x), _TS_GET_1(y))
#define _TS_PAIR_12(x, y) (_TS_GET_1(x), _TS_GET_2(y))
#define _TS_PAIR_20(x, y) (_TS_GET_2(x), _TS_GET_0(y))
#define _TS_PAIR_21(x, y) (_TS_GET_2(x), _TS_GET_1(y))
#define _TS_PAIR_22(x, y) (_TS_GET_2(x), _TS_GET_2(y))

#define _TS_COMB_11(x, y) _TS_PAIR_00(x, y)
#define _TS_COMB_12(x, y) _TS_PAIR_00(x, y), _TS_PAIR_01(x, y)
#define _TS_COMB_13(x, y) _TS_PAIR_00(x, y), _TS_PAIR_01(x, y), _TS_PAIR_02(x, y)

#define _TS_COMB_21(x, y) \
    _TS_PAIR_00(x, y),    \
    _TS_PAIR_10(x, y)

#define _TS_COMB_22(x, y)                 \
    _TS_PAIR_00(x, y), _TS_PAIR_01(x, y), \
    _TS_PAIR_10(x, y), _TS_PAIR_11(x, y)

#define _TS_COMB_23(x, y)                                    \
    _TS_PAIR_00(x, y), _TS_PAIR_01(x, y), _TS_PAIR_02(x, y), \
    _TS_PAIR_10(x, y), _TS_PAIR_11(x, y), _TS_PAIR_12(x, y)

#define _TS_COMB_31(x, y) \
    _TS_PAIR_00(x, y),    \
    _TS_PAIR_10(x, y),    \
    _TS_PAIR_20(x, y)

#define _TS_COMB_32(x, y)                 \
    _TS_PAIR_00(x, y), _TS_PAIR_01(x, y), \
    _TS_PAIR_10(x, y), _TS_PAIR_11(x, y), \
    _TS_PAIR_20(x, y), _TS_PAIR_21(x, y)

#define _TS_COMB_33(x, y)                                    \
    _TS_PAIR_00(x, y), _TS_PAIR_01(x, y), _TS_PAIR_02(x, y), \
    _TS_PAIR_10(x, y), _TS_PAIR_11(x, y), _TS_PAIR_12(x, y), \
    _TS_PAIR_20(x, y), _TS_PAIR_21(x, y), _TS_PAIR_22(x, y)

#define _TS_COMB_(N, M, x, y) _TS_CONCAT_3(_TS_COMB_, N, M)(x, y)

#define _TS_COMB(x, y) \
    _TS_COMB_(_TS_NARGS(_TS_UNPACK(x)), _TS_NARGS(_TS_UNPACK(y)), x, y)
