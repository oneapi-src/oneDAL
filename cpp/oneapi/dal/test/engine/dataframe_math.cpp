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

#include "oneapi/dal/test/engine/dataframe_math.hpp"

namespace oneapi::dal::test::engine {

template <typename Float>
inline void kahan_add(Float input, Float& sum_accumulator, Float& compensation_accumulator) {
    const Float y = input - compensation_accumulator;
    const Float t = sum_accumulator + y;
    compensation_accumulator = (t - sum_accumulator) - y;
    sum_accumulator = t;
}

template <typename Float>
static array<Float> compute_column_means(const dataframe& df) {
    const std::int64_t row_count = df.get_row_count();
    const std::int64_t column_count = df.get_column_count();

    const auto sums = array<Float>::zeros(column_count);
    const auto sums_c = array<Float>::zeros(column_count);

    const float* data = df.get_array().get_data();
    Float* sums_ptr = sums.get_mutable_data();
    Float* sums_c_ptr = sums_c.get_mutable_data();

    for (std::int64_t i = 0; i < row_count; i++) {
        for (std::int64_t j = 0; j < column_count; j++) {
            const Float x = Float(data[i * column_count + j]);
            kahan_add(x, sums_ptr[j], sums_c_ptr[j]);
        }
    }

    for (std::int64_t j = 0; j < column_count; j++) {
        sums_ptr[j] /= Float(row_count);
    }

    return sums;
}

template <typename Float>
static array<Float> compute_column_variances(const dataframe& df, const array<Float>& means) {
    const std::int64_t row_count = df.get_row_count();
    const std::int64_t column_count = df.get_column_count();

    const auto sums = array<Float>::zeros(column_count);
    const auto sums_c = array<Float>::zeros(column_count);

    const float* data = df.get_array().get_data();
    const Float* means_ptr = means.get_data();
    Float* sums_ptr = sums.get_mutable_data();
    Float* sums_c_ptr = sums_c.get_mutable_data();

    for (std::int64_t i = 0; i < row_count; i++) {
        for (std::int64_t j = 0; j < column_count; j++) {
            const Float x = Float(data[i * column_count + j]);
            const Float s = (x - means_ptr[j]) * (x - means_ptr[j]);
            kahan_add(s, sums_ptr[j], sums_c_ptr[j]);
        }
    }

    for (std::int64_t j = 0; j < column_count; j++) {
        sums_ptr[j] /= Float(row_count - 1);
    }

    return sums;
}

template <typename Float>
basic_statistics<Float> compute_basic_statistics(const dataframe& df) {
    const std::string means_key = "__cache_column_means";
    const std::string variances_key = "__cache_column_variances";

    const auto means = df.get_or_add_field<array<Float>>(means_key, [&]() {
        return compute_column_means<Float>(df);
    });

    const auto variances = df.get_or_add_field<array<Float>>(variances_key, [&]() {
        return compute_column_variances<Float>(df, means);
    });

    return { means, variances };
}

#define INSTANTIATE(Float) \
    template basic_statistics<Float> compute_basic_statistics<Float>(const dataframe& df);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::test::engine
