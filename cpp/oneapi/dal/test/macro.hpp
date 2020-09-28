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

#define _TS_STRINGIFY(x) #x
#define _TS_EXPAND(...) __VA_ARGS__
#define _TS_UNPACK(x) _TS_EXPAND x
#define _TS_CONCAT_2(_1, _2) _1##_2
#define _TS_CONCAT_3(_1, _2, _3) _1##_2##_3

#define _TS_NARGS_GET(_1, _2, _3, _4, _5, N, ...) N
#define _TS_NARGS(...) _TS_NARGS_GET(__VA_ARGS__, 5, 4, 3, 2, 1, 0)

#define _TS_FOR_EACH_0(ctx, ...)
#define _TS_FOR_EACH_1(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_0(ctx, a)
#define _TS_FOR_EACH_2(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_1(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_3(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_2(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_4(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_3(ctx, a, __VA_ARGS__)
#define _TS_FOR_EACH_5(ctx, a, x, ...) a(ctx, x) _TS_FOR_EACH_4(ctx, a, __VA_ARGS__)

#define _TS_FOR_EACH_(N, ctx, action, ...) \
    _TS_CONCAT_2(_TS_FOR_EACH_, N)(ctx, action, __VA_ARGS__)

#define _TS_FOR_EACH(ctx, action, ...) \
    _TS_FOR_EACH_(_TS_NARGS(__VA_ARGS__), ctx, action, __VA_ARGS__)

#define _TS_GET_0_(_0, ...) _0
#define _TS_GET_0(x) _TS_EXPAND(_TS_GET_0_ x)

#define _TS_GET_1_(_0, _1, ...) _1
#define _TS_GET_1(x) _TS_EXPAND(_TS_GET_1_ x)

#define _TS_GET_2_(_0, _1, _2, ...) _2
#define _TS_GET_2(x) _TS_EXPAND(_TS_GET_2_ x)

#define _TS_GET_00(x, y) (_TS_GET_0(x), _TS_GET_0(y))
#define _TS_GET_01(x, y) (_TS_GET_0(x), _TS_GET_1(y))
#define _TS_GET_02(x, y) (_TS_GET_0(x), _TS_GET_2(y))
#define _TS_GET_10(x, y) (_TS_GET_1(x), _TS_GET_0(y))
#define _TS_GET_11(x, y) (_TS_GET_1(x), _TS_GET_1(y))
#define _TS_GET_12(x, y) (_TS_GET_1(x), _TS_GET_2(y))
#define _TS_GET_20(x, y) (_TS_GET_2(x), _TS_GET_0(y))
#define _TS_GET_21(x, y) (_TS_GET_2(x), _TS_GET_1(y))
#define _TS_GET_22(x, y) (_TS_GET_2(x), _TS_GET_2(y))

#define _TS_COMB_11(x, y) _TS_GET_00(x, y)
#define _TS_COMB_12(x, y) _TS_GET_00(x, y), _TS_GET_01(x, y)
#define _TS_COMB_13(x, y) _TS_GET_00(x, y), _TS_GET_01(x, y), _TS_GET_02(x, y)

#define _TS_COMB_21(x, y) \
    _TS_GET_00(x, y), \
    _TS_GET_10(x, y)

#define _TS_COMB_22(x, y) \
    _TS_GET_00(x, y), _TS_GET_01(x, y), \
    _TS_GET_10(x, y), _TS_GET_11(x, y)

#define _TS_COMB_23(x, y) \
    _TS_GET_00(x, y), _TS_GET_01(x, y), _TS_GET_02(x, y), \
    _TS_GET_10(x, y), _TS_GET_11(x, y), _TS_GET_12(x, y)

#define _TS_COMB_31(x, y) \
    _TS_GET_00(x, y), \
    _TS_GET_10(x, y), \
    _TS_GET_20(x, y)

#define _TS_COMB_32(x, y) \
    _TS_GET_00(x, y), _TS_GET_01(x, y), \
    _TS_GET_10(x, y), _TS_GET_11(x, y), \
    _TS_GET_20(x, y), _TS_GET_21(x, y)

#define _TS_COMB_33(x, y) \
    _TS_GET_00(x, y), _TS_GET_01(x, y), _TS_GET_02(x, y), \
    _TS_GET_10(x, y), _TS_GET_11(x, y), _TS_GET_12(x, y), \
    _TS_GET_20(x, y), _TS_GET_21(x, y), _TS_GET_22(x, y)

#define _TS_COMB_(N, M, x, y) \
    _TS_CONCAT_3(_TS_COMB_, N, M)(x, y)

#define _TS_COMB(x, y) \
    _TS_COMB_(_TS_NARGS(_TS_UNPACK(x)), \
              _TS_NARGS(_TS_UNPACK(y)), x, y)
