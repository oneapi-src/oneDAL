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

#include "oneapi/dal/exceptions.hpp"

template <typename Data>
void check_sum_overflow(const Data& first, const Data& second) {
    static_assert(std::is_integral_v<Data>, "The check requires integral operands");

    volatile Data tmp = first + second;
    tmp -= first;
    if (tmp != second) {
        throw range_error("overflow found in sum of two values");
    }
}

template <typename Data>
void check_mul_overflow(const Data& first, const Data& second) {
    static_assert(std::is_integral_v<Data>, "The check requires integral operands");

    if (first != 0 && second != 0) {
        volatile Data tmp = first * second;
        tmp /= first;
        if (tmp != second) {
            throw range_error("overflow found in multiplication of two values");
        }
    }
}