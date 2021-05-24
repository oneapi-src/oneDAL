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

#include <bitset>
#include <cstdint>
#include <iostream>

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename T>
void pr(char* msg, const T& val) {
    // std::cout << msg << " " << val << std::endl;
}

#define ___PR___(x) pr(#x, x);

void ___PR8___(const std::uint8_t* arr, int n);

template <typename T>
void pr(char* msg, const T* arr, int n) {
    // std::cout << msg << "[" << n << "] : ";
    // for (int i = 0; i < n; i++) {
    //     std::cout << arr[i] << " ";
    // }
    // std::cout << std::endl;
}
#define ___PR_ARR___(x, n) pr(#x, x, n);

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
