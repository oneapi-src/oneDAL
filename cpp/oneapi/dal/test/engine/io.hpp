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

#include <iomanip>
#include <ostream>

#include "oneapi/dal/test/engine/linalg/matrix.hpp"

namespace oneapi::dal::test::engine::linalg {

template <typename T, layout lyt>
class matrix_formatter {
public:
    matrix_formatter(const matrix<T, lyt>& m) : matrix_(m) {}

    const matrix<T, lyt>& get_matrix() const {
        return matrix_;
    }

    int get_symbol_width() const {
        return symbol_width_;
    }

    auto& set_symbol_width(int symbol_width) {
        symbol_width_ = symbol_width;
        return *this;
    }

    int get_precision() const {
        return precision_;
    }

    auto& set_precision(int precision) {
        precision_ = precision;
        return *this;
    }

    std::ios::fmtflags get_iosflags() const {
        return iosflags_;
    }

    auto& set_iosflags(std::ios::fmtflags flags) {
        iosflags_ = flags;
        return *this;
    }

private:
    matrix<T, lyt> matrix_;
    int symbol_width_ = 10;
    int precision_ = 3;
    std::ios::fmtflags iosflags_ = std::ios::fixed;
};

template <typename T, layout lyt>
inline matrix_formatter<T, lyt> format_matrix(const matrix<T, lyt>& m) {
    return matrix_formatter<T, lyt>{ m };
}

template <typename T, layout lyt>
inline std::ostream& operator<<(std::ostream& stream, const matrix_formatter<T, lyt>& mf) {
    const auto& m = mf.get_matrix();
    for (std::int64_t i = 0; i < m.get_row_count(); i++) {
        for (std::int64_t j = 0; j < m.get_column_count(); j++) {
            stream << std::setw(mf.get_symbol_width());
            stream << std::setprecision(mf.get_precision());
            stream << std::setiosflags(mf.get_iosflags());
            stream << m.get(i, j);
            if (j + 1 == m.get_column_count()) {
                stream << std::endl;
            }
        }
    }
    return stream;
}

template <typename T, layout lyt>
inline std::ostream& operator<<(std::ostream& stream, const matrix<T, lyt>& m) {
    return stream << format_matrix(m);
}

inline std::ostream& operator<<(std::ostream& stream, layout l) {
    if (l == layout::row_major) {
        stream << "row-major";
    }
    else {
        stream << "column-major";
    }
    return stream;
}

} // namespace oneapi::dal::test::engine::linalg
