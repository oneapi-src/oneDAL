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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

#define _P(...)              \
    do {                     \
        printf(__VA_ARGS__); \
        printf("\n");        \
        fflush(0);           \
    } while (0)

#define _PL(...)             \
    do {                     \
        printf(__VA_ARGS__); \
        fflush(0);           \
    } while (0)

/// Performs inplace radix sort of input vector and corresponding indices
///
/// @tparam Float Floating-point type used for storing input values
/// @tparam IndexType Integer type used for storing input indices
///
/// @param[in]  queue The queue
/// @param[in|out]  val  The [n] input/output vector of values to sort out
/// @param[in|out]  ind  The [n] input/output vector of corresponding indices
/// @param[in]  val_buff The [n] auxiliary buff for storing values
/// @param[in]  ind_buff The [n] auxiliary buff for storing indices
template <typename Float, typename IndexType = std::uint32_t>
sycl::event radix_sort_indices_inplace(sycl::queue& queue,
                                       ndview<Float, 1>& val,
                                       ndview<IndexType, 1>& ind,
                                       ndview<Float, 1>& val_buff,
                                       ndview<IndexType, 1>& ind_buff,
                                       const event_vector& deps = {});

/// Performs radix sort of batch of integer input vectors
/// NOTE: only positive values are supported for now
///
/// @tparam IntType Integer type used for storing input values
///
/// @param[in]  queue The queue
/// @param[in]  val_in The [n x p] input array of vectors (row major format) to sort out
/// @param[out] val_out The [n x p] output array of sorted vectors
/// @param[in]  buffer The [n x 256] array of auxiliary buffer
/// @param[in]  sorted_elem_count The number of elements to sort in each vector
template <typename IntType>
sycl::event radix_sort(sycl::queue& queue,
                       ndview<IntType, 2>& val_in,
                       ndview<IntType, 2>& val_out,
                       ndview<IntType, 2>& buffer,
                       std::uint32_t sorted_elem_count,
                       const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
