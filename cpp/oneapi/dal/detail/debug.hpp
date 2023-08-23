/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include <iostream>

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::detail {

#ifdef _GLIBCXX_OSTREAM

template <typename Type>
struct print_type {
    using type = Type;
};

template <typename Type>
struct print_type<Type*> {
    using type = const void*;
};

template <typename Type>
using print_type_t = typename print_type<Type>::type;

template <typename Type>
inline auto& print_array_shape(std::ostream& s, const array<Type>& a) {
    const auto c = a.get_count();

    if (c == 0)
        return s << "Empty array \n";

    return s << "Array of size " << c << "\n";
}

inline auto& print_table_shape(std::ostream& s, const table& t) {
    const auto h = t.get_row_count();
    const auto w = t.get_column_count();

    if (h * w == 0)
        return s << "Empty table \n";

    return s << "Table with shape height,width=" << h << ',' << w << "\n";
}

template <typename Type, typename PrintType = print_type_t<Type>>
inline auto& print_array_content(std::ostream& s, const array<Type>& arr) {
    const auto c = arr.get_count();

#ifdef _GLIBCXX_IOMANIP
    const auto init_flags = s.flags();
    s << std::scientific << std::setprecision(4);
#endif

    for (std::int64_t i = 0; i < c; ++i) {
        s << "\t " << static_cast<PrintType>(arr[i]);
    }

#ifdef _GLIBCXX_IOMANIP
    s.setf(init_flags);
#endif

    return s;
}

template <typename Float = float>
inline auto& print_table_content(std::ostream& s, const table& t) {
    [[maybe_unused]] const auto w = t.get_column_count();
    const auto h = t.get_row_count();

    row_accessor<const Float> accessor(t);
    for (std::int64_t r = 0; r < h; ++r) {
        auto row = accessor.pull({ r, r + 1 });
        ONEDAL_ASSERT(w == row.get_count());
        print_array_content(s, row);
        s << "\t: r" << r << '\n';
    }

    return s;
}

template <typename Type, typename PrintType = print_type_t<Type>>
inline std::ostream& operator<<(std::ostream& s, const array<Type>& arr) {
    print_array_shape(s, arr);
    print_array_content<Type, PrintType>(s, arr);
    return s << std::endl;
}

template <typename Float = float>
inline std::ostream& operator<<(std::ostream& s, const table& t) {
    print_table_shape(s, t);
    print_table_content<Float>(s, t);
    return s << std::endl;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& s, const std::pair<T1, T2>& p) {
    return s << '{' << p.first << ',' << p.second << '}';
}

#endif

} // namespace oneapi::dal::detail
