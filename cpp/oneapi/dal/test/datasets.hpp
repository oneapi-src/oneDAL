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

#include <string>

#include "oneapi/dal/test/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::test {

#define GENERATE_DATASET(...) \
    GENERATE(as<oneapi::dal::test::dataset_builder>{}, __VA_ARGS__).build()

#define ITERATE_OVER_TABLE_TYPES(var_name, ...)                            \
    const std::string var_name = GENERATE(as<std::string>{}, __VA_ARGS__); \
    SECTION("iterate over table types: " + var_name)

class dataset {
public:
    explicit dataset(const array<float>& data, std::int64_t row_count, std::int64_t column_count);

    template <typename Float>
    table get_table(const std::string& table_type) const;

    template <typename Float>
    table get_table(host_test_policy& policy, const std::string& table_type) const {
        return get_table<Float>(table_type);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Float>
    table get_table(device_test_policy& policy, const std::string& table_type) const;
#endif

private:
    class impl;

    explicit dataset(impl* i);

    const impl& get() const {
        return *impl_;
    }

    dal::detail::pimpl<impl> impl_;
};

enum class distribution_type { constant, uniform, normal };

class dataset_generator {
public:
    std::int64_t row_count_;
    std::int64_t column_count_;
    distribution_type distribution_ = distribution_type::uniform;
    std::int64_t seed_ = 7777;
    double uniform_a_ = 0.0;
    double uniform_b_ = 1.0;

    dataset_generator(std::int64_t row_count, std::int64_t column_count)
            : row_count_(row_count),
              column_count_(column_count) {}

    dataset generate() const;
};

class dataset_builder {
public:
    dataset build() const {
        return impl_->build();
    }

protected:
    class impl {
    public:
        virtual ~impl() = default;
        virtual dataset build() = 0;
    };

    explicit dataset_builder(impl* i) : impl_(i) {}

    template <typename T>
    T& get() {
        return static_cast<T&>(*impl_);
    }

    template <typename T>
    const T& get() const {
        return static_cast<T&>(*impl_);
    }

private:
    dal::detail::pimpl<impl> impl_;
};

class random_dataset : public dataset_builder {
public:
    explicit random_dataset(std::int64_t row_count, std::int64_t column_count)
            : dataset_builder(new impl{ row_count, column_count }) {}

    auto& uniform(double a, double b) {
        get<impl>().distribution_ = distribution_type::uniform;
        get<impl>().uniform_a_ = a;
        get<impl>().uniform_b_ = b;
        return *this;
    }

    auto& seed(std::int64_t seed) {
        get<impl>().seed_ = seed;
        return *this;
    }

protected:
    class impl : public dataset_builder::impl, public dataset_generator {
    public:
        impl(std::int64_t row_count, std::int64_t column_count)
                : dataset_generator(row_count, column_count) {}

        dataset build() override {
            return dataset_generator::generate();
        }
    };
};

} // namespace oneapi::dal::test
