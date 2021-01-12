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

#include "oneapi/dal/test/engine/datasets.hpp"

#include <regex>
#include <memory>
#include <random>
#include <sstream>
#include <iostream>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::test::engine {

class dataset::impl {
public:
    explicit impl(const array<float>& data, std::int64_t row_count, std::int64_t column_count)
            : data_(data),
              row_count_(row_count),
              column_count_(column_count) {}

    const array<float>& get_data() const {
        return data_;
    }

    std::int64_t get_row_count() const {
        return row_count_;
    }

    std::int64_t get_column_count() const {
        return column_count_;
    }

private:
    array<float> data_;
    std::int64_t row_count_;
    std::int64_t column_count_;
};

dataset::dataset(const array<float>& data, std::int64_t row_count, std::int64_t column_count)
        : dataset(new impl{ data, row_count, column_count }) {}

static std::mt19937& get_random_engine() {
    static std::mt19937 engine(77777);
    return engine;
}

static void generate_uniform(array<float>& data, double a, double b) {
    auto& engine = get_random_engine();
    std::uniform_real_distribution<float> distr(a, b);
    float* data_raw = data.get_mutable_data();
    for (std::int64_t i = 0; i < data.get_count(); i++) {
        data_raw[i] = distr(engine);
    }
}

dataset dataset_generator::generate() const {
    const std::int64_t element_count = row_count_ * column_count_;
    auto data = array<float>::empty(element_count);
    switch (distribution_) {
        case distribution_type::uniform: generate_uniform(data, uniform_a_, uniform_b_); break;
        default:
            throw unimplemented{
                "Only uniform distribution is implemented now in dataset generator"
            };
    }
    return dataset{ data, row_count_, column_count_ };
}

dataset::dataset(impl* i) : impl_(i) {}

template <typename Float>
table dataset::get_table(const std::string& table_type) const {
    const auto data = get().get_data();
    const std::int64_t row_count = get().get_row_count();
    const std::int64_t column_count = get().get_column_count();
    if (table_type == "homogen") {
        return dal::detail::homogen_table_builder{}.reset(data, row_count, column_count).build();
    }
    else {
        throw unimplemented{ dal::detail::error_messages::only_homogen_table_is_supported() };
    }
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Float>
table dataset::get_table(device_test_policy& policy, const std::string& table_type) const {
    const auto data = get().get_data();
    const std::int64_t row_count = get().get_row_count();
    const std::int64_t column_count = get().get_column_count();
    if (table_type == "homogen") {
        return dal::detail::homogen_table_builder{}
            .allocate(policy.get_queue(), row_count, column_count)
            .copy_data(policy.get_queue(), data.get_data(), row_count, column_count)
            .build();
    }
    else {
        throw unimplemented{ dal::detail::error_messages::only_homogen_table_is_supported() };
    }
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
#define INSTANTIATE(F)                                              \
    template table dataset::get_table<F>(const std::string&) const; \
    template table dataset::get_table<F>(device_test_policy&, const std::string&) const;
#else
#define INSTANTIATE(F) template table dataset::get_table<F>(const std::string&) const;
#endif

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::test::engine
