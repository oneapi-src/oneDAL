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

#include <cstdint>

#include "oneapi/dal/detail/serialization.hpp"

namespace oneapi::dal {

class result_option_id_base {
    using this_t = result_option_id_base;
    friend dal::detail::serialization_accessor;

public:
    using bitset_t = std::uint64_t;
    constexpr result_option_id_base() : mask_{ bitset_t(0) } {}
    constexpr explicit result_option_id_base(const bitset_t& mask) : mask_{ mask } {}

    constexpr operator bool() const {
        return mask_ > 0;
    }

    constexpr const bitset_t& get_mask() const {
        return mask_;
    }

    constexpr bool test(const result_option_id_base& flag) const {
        return get_mask() & flag.get_mask();
    }

    constexpr static this_t make_by_index(std::int64_t result_index) {
        return this_t{ bitset_t(1) << result_index };
    }

private:
    void serialize(detail::output_archive& ar) const {
        ar(mask_);
    }

    void deserialize(detail::input_archive& ar) {
        ar(mask_);
    }

    bitset_t mask_;
};

template <typename ResultOptionIdType>
constexpr inline bool is_result_option_id_v =
    std::is_base_of_v<result_option_id_base, ResultOptionIdType>;

template <typename ResultOptionIdType>
using enable_if_result_option_id_t = std::enable_if_t<is_result_option_id_v<ResultOptionIdType>>;

template <typename ResultOptionIdType, typename = enable_if_result_option_id_t<ResultOptionIdType>>
constexpr inline ResultOptionIdType operator|(const ResultOptionIdType& lhs,
                                              const ResultOptionIdType& rhs) {
    return ResultOptionIdType{ result_option_id_base{ lhs.get_mask() | rhs.get_mask() } };
}

template <typename ResultOptionIdType, typename = enable_if_result_option_id_t<ResultOptionIdType>>
constexpr inline ResultOptionIdType operator&(const ResultOptionIdType& lhs,
                                              const ResultOptionIdType& rhs) {
    return ResultOptionIdType{ result_option_id_base{ lhs.get_mask() & rhs.get_mask() } };
}

template <typename ResultOptionIdType, typename = enable_if_result_option_id_t<ResultOptionIdType>>
constexpr inline ResultOptionIdType operator~(const ResultOptionIdType& hs) {
    return ResultOptionIdType{ result_option_id_base{ ~(hs.get_mask()) } };
}

template <typename ResultOptionIdType, typename = enable_if_result_option_id_t<ResultOptionIdType>>
constexpr inline bool operator==(const ResultOptionIdType& lhs, const ResultOptionIdType& rhs) {
    return lhs.get_mask() == rhs.get_mask();
}

template <typename ResultOptionIdType, typename = enable_if_result_option_id_t<ResultOptionIdType>>
constexpr inline bool operator!=(const ResultOptionIdType& lhs, const ResultOptionIdType& rhs) {
    return lhs.get_mask() != rhs.get_mask();
}

} // namespace oneapi::dal
