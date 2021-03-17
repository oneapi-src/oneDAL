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

#include "oneapi/dal/backend/primitives/selection/select_by_rows.hpp"
#include "oneapi/dal/backend/primitives/selection/select_by_rows_simd.hpp"
#include "oneapi/dal/backend/primitives/selection/select_by_rows_quick.hpp"

namespace oneapi::dal::backend::primitives {

constexpr uint32_t simd16 = 16;
constexpr uint32_t simd32 = 32;
constexpr uint32_t simd64 = 64;
constexpr uint32_t simd128 = 128;

/* Commented params are for kNN perfomance improvement (optimized usage of GEMM) */

template <typename Float, bool selection_out, bool indices_out>
sycl::event select_by_rows_impl(sycl::queue& queue,
                                const ndview<Float, 2>& data,
                                /*const ndview<Float, 1>& add_by_col,*/
                                std::int64_t k,
                                std::int64_t col_begin,
                                std::int64_t col_end,
                                ndview<Float, 2>& selection,
                                ndview<int, 2>& column_indices,
                                const event_vector& deps) {
    if constexpr (selection_out) {
        ONEDAL_ASSERT(data.get_dimension(1) == selection.get_dimension(1));
        ONEDAL_ASSERT(data.get_dimension(0) >= k);
        ONEDAL_ASSERT(selection.get_dimension(0) == k);
        ONEDAL_ASSERT(selection.has_mutable_data());
    }
    if constexpr (indices_out) {
        ONEDAL_ASSERT(data.get_dimension(1) == indices.get_dimension(1));
        ONEDAL_ASSERT(indices.get_dimension(0) == k);
        ONEDAL_ASSERT(indices.has_mutable_data());
    }

    const auto sg_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
    ONEDAL_ASSERT(!sg_sizes.empty());
    auto result = std::max_element(sg_sizes.begin(), sg_sizes.end());
    ONEDAL_ASSERT(result != sg_sizes.end());
    const uint32_t simd_width = static_cast<int>(*result);

    if (k <= simd_width) {
        if (simd_width == simd16) {
            return select_by_rows_simd<Float, simd16, selection_out, indices_out>(
                queue,
                data,
                /*                                                                                add_by_col,*/
                k,
                col_begin,
                col_end,
                selection,
                column_indices,
                deps);
        }
        if (simd_width == simd32) {
            return select_by_rows_simd<Float, simd32, selection_out, indices_out>(
                queue,
                data,
                /*                                                                                add_by_col,*/
                k,
                col_begin,
                col_end,
                selection,
                column_indices,
                deps);
        }
        if (simd_width == simd64) {
            return select_by_rows_simd<Float, simd64, selection_out, indices_out>(
                queue,
                data,
                /*                                                                                add_by_col,*/
                k,
                col_begin,
                col_end,
                selection,
                column_indices,
                deps);
        }
        if (simd_width == simd128) {
            return select_by_rows_simd<Float, simd128, selection_out, indices_out>(
                queue,
                data,
                /*                                                                                add_by_col,*/
                k,
                col_begin,
                col_end,
                selection,
                column_indices,
                deps);
        }
        ONEDAL_ASSERT(false);
        return sycl::event();
    }
    else {
        return select_by_rows_quick<Float, selection_out, indices_out>(
            queue,
            data,
            /*                                                                                add_by_col,*/
            k,
            col_begin,
            col_end,
            selection,
            column_indices,
            deps);
    }
}

template <typename Float>
sycl::event selection_by_rows<Float>::select(sycl::queue& queue,
                                             std::int64_t k,
                                             ndview<Float, 2>& selection,
                                             ndview<int, 2>& column_indices,
                                             const event_vector& deps) {
    return select_by_rows_impl<Float, true, true>(queue,
                                                  this->data_,
                                                  /*this->add_by_col_,*/ k,
                                                  this->col_begin_,
                                                  this->col_end_,
                                                  selection,
                                                  column_indices,
                                                  deps);
}

template <typename Float>
sycl::event selection_by_rows<Float>::select(sycl::queue& queue,
                                             std::int64_t k,
                                             ndview<Float, 2>& selection,
                                             const event_vector& deps) {
    ndarray<int, 2> dummy_array;
    return select_by_rows_impl<Float, true, false>(queue,
                                                   this->data_,
                                                   /*this->add_by_col_,*/ k,
                                                   this->col_begin_,
                                                   this->col_end_,
                                                   selection,
                                                   dummy_array,
                                                   deps);
}

template <typename Float>
sycl::event selection_by_rows<Float>::select(sycl::queue& queue,
                                             std::int64_t k,
                                             ndview<int, 2>& column_indices,
                                             const event_vector& deps) {
    ndarray<Float, 2> dummy_array;
    return select_by_rows_impl<Float, false, true>(queue,
                                                   this->data_,
                                                   /*this->add_by_col_,*/ k,
                                                   this->col_begin_,
                                                   this->col_end_,
                                                   dummy_array,
                                                   column_indices,
                                                   deps);
}

#define INSTANTIATE_IMPL(F, selection_out, indices_out)                                    \
    template ONEDAL_EXPORT sycl::event select_by_rows_impl<F, selection_out, indices_out>( \
        sycl::queue & queue,                                                               \
        const ndview<F, 2>& data, /*const ndview<F, 1>& add_by_col,*/                      \
        std::int64_t k,                                                                    \
        std::int64_t col_begin,                                                            \
        std::int64_t col_end,                                                              \
        ndview<F, 2>& selection,                                                           \
        ndview<int, 2>& column_indices,                                                    \
        const event_vector& deps);

#define INSTANTIATE_IMPL_FLOAT(selection_out, indices_out) \
    INSTANTIATE_IMPL(float, selection_out, indices_out)    \
    INSTANTIATE_IMPL(double, selection_out, indices_out)

INSTANTIATE_IMPL_FLOAT(true, false)
INSTANTIATE_IMPL_FLOAT(false, true)
INSTANTIATE_IMPL_FLOAT(true, true)

#define INSTANTIATE(F) template ONEDAL_EXPORT class selection_by_rows<F>;

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::primitives
