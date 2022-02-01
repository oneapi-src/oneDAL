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

#include <iostream>

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef _GLIBCXX_OSTREAM

// Available only if (i)ostream header is included
template <typename T, ndorder ord>
inline std::ostream& operator<<(std::ostream& s, const ndview<T, 2, ord>& v) {
    constexpr char o = (ord == ndorder::c) ? 'C' : 'F';
    const auto h = v.get_dimension(0);
    const auto w = v.get_dimension(1);
    const auto d = v.get_leading_stride();
    s << o << "-like ndview with shape height,width=" << h << ',' << w << " (stride=" << d << ")\n";
#ifdef _GLIBCXX_IOMANIP
    const auto init_flags = s.flags();
    s << std::scientific << std::setprecision(4);
#endif
    for (std::int64_t r = 0; r < h; ++r) {
        for (std::int64_t c = 0; c < w; ++c) {
            s << "\t " << v.at(r, c);
        }
        s << "\t: r" << r << '\n';
    }
#ifdef _GLIBCXX_IOMANIP
    s.setf(init_flags);
#endif
    return (s << std::endl);
}

#endif

} // namespace oneapi::dal::backend::primitives
