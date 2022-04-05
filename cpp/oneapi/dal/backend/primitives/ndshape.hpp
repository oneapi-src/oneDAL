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

#include <array>

#include "oneapi/dal/backend/primitives/common.hpp"

namespace oneapi::dal::backend::primitives {

template <std::int64_t axis_count>
class ndshape {
public:
    ndshape() {
        dimensions_.fill(0);
    }

    ndshape(const ndindex<axis_count>& dimensions) : dimensions_(dimensions) {
        for (std::int64_t i = 0; i < axis_count; i++) {
            ONEDAL_ASSERT(dimensions[i] > 0, "Dimension must be positive number");
        }

#ifdef ONEDAL_ENABLE_ASSERT
        std::int64_t product = 1;
        for (std::int64_t i = 0; i < axis_count; i++) {
            ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, product, dimensions[i]);
            product *= dimensions[i];
        }
#endif
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    ndshape(std::int64_t d1) : ndshape(ndindex<1>{ d1 }) {}

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    ndshape(std::int64_t d1, std::int64_t d2) : ndshape(ndindex<2>{ d1, d2 }) {}

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 3>>
    ndshape(std::int64_t d1, std::int64_t d2, std::int64_t d3)
            : ndshape(ndindex<3>{ d1, d2, d3 }) {}

    static ndshape square(std::int64_t dimension) {
        ndindex<axis_count> dimensions;
        dimensions.fill(dimension);
        return ndshape{ dimensions };
    }

    std::int64_t operator[](std::int64_t i) const {
        return this->at(i);
    }

    std::int64_t at(std::int64_t i) const {
        ONEDAL_ASSERT(i < axis_count, "Index is out of range");
        return dimensions_[i];
    }

    std::int64_t get_count() const {
        std::int64_t product = 1;
        for (std::int64_t i = 0; i < axis_count; i++) {
            // Multiplication overflow is checked in the constructor
            product *= dimensions_[i];
        }
        return product;
    }

    ndshape t() const {
        ndindex<axis_count> dimensions_reversed;
        for (std::int64_t i = 0; i < axis_count; i++) {
            dimensions_reversed[i] = dimensions_[axis_count - i - 1];
        }
        return ndshape<axis_count>{ dimensions_reversed };
    }

    bool operator==(const ndshape& other) const {
        bool is_equal = true;
        for (std::int64_t i = 0; i < axis_count; i++) {
            is_equal = is_equal && (dimensions_[i] == other[i]);
        }
        return is_equal;
    }

    bool operator!=(const ndshape& other) const {
        return !(*this == other);
    }

    const ndindex<axis_count>& get_index() const {
        return dimensions_;
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    sycl::range<1> to_range() const {
        return make_range_1d(at(0));
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    sycl::range<2> to_range() const {
        return make_range_2d(at(0), at(1));
    }
#endif

private:
    ndindex<axis_count> dimensions_;
};

} // namespace oneapi::dal::backend::primitives
