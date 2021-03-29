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

constexpr std::uint32_t simd16 = 16;
constexpr std::uint32_t simd32 = 32;
constexpr std::uint32_t simd64 = 64;
constexpr std::uint32_t simd128 = 128;

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
select_by_rows<Float>::select_by_rows(sycl::queue& queue, const ndshape<2>& shape, std::int64_t k) {
    const auto sg_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
    ONEDAL_ASSERT(!sg_sizes.empty());
    auto result = std::max_element(sg_sizes.begin(), sg_sizes.end());
    ONEDAL_ASSERT(result != sg_sizes.end());
    const std::uint32_t simd_width = static_cast<std::uint32_t>(*result);

    using base_ptr = detail::unique<select_by_rows_base<Float>>;

    if (k <= simd_width) {
        if (simd_width == simd16) {
            base_ = std::move(base_ptr(new select_by_rows_simd<Float, simd16>()));
            return;
        }
        else if (simd_width == simd32) {
            base_ = std::move(base_ptr(new select_by_rows_simd<Float, simd32>()));
            return;
        }
        else if (simd_width == simd64) {
            base_ = std::move(base_ptr(new select_by_rows_simd<Float, simd64>()));
            return;
        }
        else if (simd_width == simd128) {
            base_ = std::move(base_ptr(new select_by_rows_simd<Float, simd128>()));
            return;
        }
        ONEDAL_ASSERT(false);
    }
    else {
        base_ = std::move(base_ptr(new select_by_rows_quick<Float>(queue, shape)));
    }
}

#define INSTANTIATE(F) template class select_by_rows<F>;

INSTANTIATE(float)
INSTANTIATE(double)

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
