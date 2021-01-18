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
#include <vector>
#include <memory>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test::engine {

#define GENERATE_DATAFRAME(...) \
    GENERATE(as<oneapi::dal::test::engine::dataframe_builder>{}, __VA_ARGS__).build()

class dataframe_impl {
public:
    explicit dataframe_impl(const array<float>& data,
                            std::int64_t row_count,
                            std::int64_t column_count)
            : array_(data),
              row_count_(row_count),
              column_count_(column_count) {
        array_.need_mutable_data();
    }

    dataframe_impl(const dataframe_impl&) = delete;
    dataframe_impl& operator=(const dataframe_impl&) = delete;

    std::int64_t get_count() const {
        return array_.get_count();
    }

    std::size_t get_size() const {
        return array_.get_size();
    }

    std::int64_t get_row_count() const {
        return row_count_;
    }

    std::int64_t get_column_count() const {
        return column_count_;
    }

    float* get_data() const {
        return array_.get_mutable_data();
    }

    const array<float>& get_array() const {
        return array_;
    }

    dataframe_impl* copy() const {
        auto array_copy = array<float>::empty(array_.get_count());
        float* array_copy_data = array_copy.get_mutable_data();
        const float* array_data = array_.get_data();
        for (std::int64_t i = 0; i < array_.get_count(); i++) {
            array_copy_data[i] = array_data[i];
        }
        return new dataframe_impl{ array_copy, row_count_, column_count_ };
    }

private:
    array<float> array_;
    std::int64_t row_count_;
    std::int64_t column_count_;
};

class dataframe {
public:
    explicit dataframe(const array<float>& data, std::int64_t row_count, std::int64_t column_count)
            : dataframe(new dataframe_impl{ data, row_count, column_count }) {}

    explicit dataframe(dataframe_impl* impl) : impl_(impl) {}

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

    std::int64_t get_row_count() const {
        return impl_->get_row_count();
    }

    std::int64_t get_column_count() const {
        return impl_->get_column_count();
    }

    std::size_t get_count() const {
        return impl_->get_count();
    }

    std::size_t get_size() const {
        return impl_->get_size();
    }

private:
    dal::detail::pimpl<dataframe_impl> impl_;
};

class dataframe_builder_action {
public:
    virtual ~dataframe_builder_action() = default;
    virtual std::string get_opcode() const = 0;
    virtual dataframe_impl* execute(dataframe_impl* df) const = 0;
};

class dataframe_builder_program {
public:
    dataframe_builder_program() = default;
    dataframe_builder_program(const dataframe_builder_program&) = delete;
    dataframe_builder_program& operator=(const dataframe_builder_program&) = delete;

    template <typename Action, typename... Args>
    void add(Args&&... args) {
        actions_.emplace_back(new Action{ std::forward<Args>(args)... });
        code_ += actions_.back()->get_opcode();
    }

    dataframe execute() const;

    const std::string& get_code() const {
        return code_;
    }

private:
    std::string code_;
    std::vector<std::unique_ptr<dataframe_builder_action>> actions_;
};

class dataframe_builder_impl {
public:
    explicit dataframe_builder_impl(std::int64_t row_count, std::int64_t column_count);
    dataframe_builder_impl(const dataframe_builder_impl&) = delete;
    dataframe_builder_impl& operator=(const dataframe_builder_impl&) = delete;

    dataframe_builder_program& get_program() {
        return program_;
    }

private:
    dataframe_builder_program program_;
};

class dataframe_builder {
public:
    explicit dataframe_builder(std::int64_t row_count, std::int64_t column_count)
            : impl_(new dataframe_builder_impl{ row_count, column_count }) {}

    dataframe_builder& fill_uniform(double a, double b, std::int64_t seed = 7777);

    dataframe build() const;

protected:
    dal::detail::pimpl<dataframe_builder_impl> impl_;
};

} // namespace oneapi::dal::test::engine
