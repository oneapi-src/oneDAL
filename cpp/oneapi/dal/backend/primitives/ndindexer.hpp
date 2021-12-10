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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/ndshape.hpp"

namespace oneapi::dal::backend::primitives {
template <typename Index = std::int64_t>
struct ndindexer_dkeeper {
    ndindexer_dkeeper(Index h, Index w, Index s)
            : width{ std::move(w) },
              height{ std::move(h) },
              stride{ std::move(s) } {
#ifndef __SYCL_DEVICE_ONLY__
        ONEDAL_ASSERT(0 < width);
        ONEDAL_ASSERT(0 < height);
        ONEDAL_ASSERT(0 < stride);
#endif
    }

    inline void check(const Index& row, const Index& col) const {
#ifndef __SYCL_DEVICE_ONLY__
        ONEDAL_ASSERT((0 <= row) && (row < height));
        ONEDAL_ASSERT((0 <= col) && (col < width));
#endif
    }

    const Index width;
    const Index height;
    const Index stride;
};

template <ndorder order, typename Index = std::int64_t>
struct ndindexer_base : public ndindexer_dkeeper<Index> {};

template <typename Index>
struct ndindexer_base<ndorder::c, Index> : public ndindexer_dkeeper<Index> {
    using base_t = ndindexer_dkeeper<Index>;

    ndindexer_base(Index h, Index w, Index s) : base_t(std::move(h), std::move(w), std::move(s)) {
#ifndef __SYCL_DEVICE_ONLY__
        ONEDAL_ASSERT(base_t::width <= base_t::stride);
#endif
    }

    Index get_index(Index row, Index col) const {
        base_t::check(row, col);
        return row * base_t::stride + col;
    }
};

template <typename Index>
struct ndindexer_base<ndorder::f, Index> : public ndindexer_dkeeper<Index> {
    using base_t = ndindexer_dkeeper<Index>;

    ndindexer_base(Index h, Index w, Index s) : base_t(std::move(h), std::move(w), std::move(s)) {
#ifndef __SYCL_DEVICE_ONLY__
        ONEDAL_ASSERT(base_t::height <= base_t::stride);
#endif
    }

    Index get_index(Index row, Index col) const {
        base_t::check(row, col);
        return col * base_t::stride + row;
    }
};

template <typename Type, ndorder order, typename Index = std::int64_t>
struct ndindexer : public ndindexer_base<order, Index> {
    using base_t = ndindexer_base<order, Index>;

    ndindexer(const Type* const d, Index h, Index w, Index s)
            : base_t(std::move(h), std::move(w), std::move(s)),
              data{ std::move(d) } {}

    Type& at(Index row, Index col) const {
        const auto index = base_t::get_index(std::move(row), std::move(col));
        return *(get_mutable_data() + index);
    }

    // TODO: Apparently doesn't work
    /*const Type& at(Index row, Index col) const {
            const auto index = base_t::get_index(std::move(row), std::move(col));
            return *(get_data() + index);
        }*/

    const Type* get_data() const {
        return data;
    }

    Type* get_mutable_data() const {
        return const_cast<Type*>(data);
    }

    const Type* const data;
};

template <typename Type, ndorder order>
inline auto make_ndindexer(const ndview<Type, 2, order>& view) {
    return ndindexer<Type, order>(std::move(view.get_data()),
                                  std::move(view.get_dimension(0)),
                                  std::move(view.get_dimension(1)),
                                  std::move(view.get_leading_stride()));
}
} // namespace oneapi::dal::backend::primitives
