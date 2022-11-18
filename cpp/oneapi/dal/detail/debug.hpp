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

#include <iostream>

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::detail {

#ifdef _GLIBCXX_OSTREAM

inline auto& print_table_shape(std::ostream& s, const table& t) {
    const auto h = t.get_row_count();
    const auto w = t.get_column_count();

    return s << "Table with shape height,width=" << h << ',' << w << "\n";
}

template <typename Float = float>
inline auto& print_table_content(std::ostream& s, const table& t) {
    const auto h = t.get_row_count();
    const auto w = t.get_column_count();

#ifdef _GLIBCXX_IOMANIP
    const auto init_flags = s.flags();
    s << std::scientific << std::setprecision(4);
#endif

    row_accessor<const Float> accessor(t);
    for (std::int64_t r = 0; r < h; ++r) {
        auto row = accessor.pull({ r, r + 1 });
        for (std::int64_t c = 0; c < w; ++c) {
            s << "\t " << row[c];
        }
        s << "\t: r" << r << '\n';
    }
#ifdef _GLIBCXX_IOMANIP
    s.setf(init_flags);
#endif
    return s;
}

template <typename Float = float>
inline std::ostream& operator<<(std::ostream& s, const table& t) {
    print_table_shape(s, t);
    print_table_content<Float>(s, t);
    return s << std::endl;
}

#endif

} // namespace oneapi::dal::detail
