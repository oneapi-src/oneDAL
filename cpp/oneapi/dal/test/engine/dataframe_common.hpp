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

#include <any>
#include <string>
#include <vector>
#include <memory>
#include <optional>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/io/csv.hpp"

namespace oneapi::dal::test::engine {

#define GENERATE_DATAFRAME(...) \
    GENERATE_COPY(as<oneapi::dal::test::engine::dataframe_builder>{}, __VA_ARGS__).build()

enum class table_kind { homogen };

enum class table_float_type { f32, f64 };

class table_id {
public:
    template <typename Float>
    static table_id homogen() {
        static_assert(dal::detail::is_floating_point<Float>());
        if constexpr (std::is_same_v<Float, float>) {
            return table_id{ table_kind::homogen, table_float_type::f32 };
        }
        return table_id{ table_kind::homogen, table_float_type::f64 };
    }

    table_kind get_kind() const {
        return kind_;
    }

    table_float_type get_float_type() const {
        return float_type_;
    }

private:
    explicit table_id(table_kind kind, table_float_type float_type)
            : kind_(kind),
              float_type_(float_type) {}

    table_kind kind_;
    table_float_type float_type_;
};

class dataframe_impl {
public:
    explicit dataframe_impl(const array<float>& data,
                            std::int64_t row_count,
                            std::int64_t column_count)
            : array_(data),
              row_count_(row_count),
              column_count_(column_count) {}

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

    const array<float>& get_array() const {
        return array_;
    }

    array<float>& get_array() {
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

    template <typename T>
    void add_field(const std::string& name, T&& value) {
        user_fields_.insert_or_assign(name, std::any{ std::forward<T>(value) });
    }

    void remove_field(const std::string& name) {
        user_fields_.erase(name);
    }

    template <typename T>
    std::optional<T> get_field(const std::string& name) const {
        const auto it = user_fields_.find(name);
        if (it == user_fields_.end()) {
            return std::nullopt;
        }

        try {
            return std::any_cast<T>(it->second);
        }
        catch (const std::bad_any_cast&) {
            return std::nullopt;
        }
    }

private:
    array<float> array_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    std::unordered_map<std::string, std::any> user_fields_;
};

class dataframe {
public:
    explicit dataframe(const float* data, std::int64_t row_count, std::int64_t column_count)
            : dataframe(new dataframe_impl{ array<float>::wrap(data, row_count * column_count),
                                            row_count,
                                            column_count }) {}

    explicit dataframe(const array<float>& data, std::int64_t row_count, std::int64_t column_count)
            : dataframe(new dataframe_impl{ data, row_count, column_count }) {}

    explicit dataframe(dataframe_impl* impl) : impl_(impl) {}

    table get_table(host_test_policy& policy, const table_id& id, const range& r = { 0, 0 }) const;

#ifdef ONEDAL_DATA_PARALLEL
    table get_table(device_test_policy& policy,
                    const table_id& id,
                    const range& r = { 0, 0 },
                    sycl::usm::alloc alloc = sycl::usm::alloc::shared) const;

    table get_table(device_test_policy& policy, const table_id& id, sycl::usm::alloc alloc) const {
        return get_table(policy, id, { 0, 0 }, alloc);
    }
#endif

    table get_table(const table_id& id, const range& r = { 0, 0 }) const {
        host_test_policy policy;
        return get_table(policy, id, r);
    }

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

    const array<float>& get_array() const {
        return impl_->get_array();
    }

    template <typename T>
    void add_field(const std::string& name, T&& value) const {
        impl_->add_field(name, std::forward<T>(value));
    }

    void remove_field(const std::string& name) const {
        impl_->remove_field(name);
    }

    template <typename T>
    std::optional<T> get_field(const std::string& name) const {
        return impl_->template get_field<T>(name);
    }

    template <typename T, typename Op>
    T get_or_add_field(const std::string& name, Op&& op) const {
        const auto optional_value = impl_->template get_field<T>(name);
        if (optional_value) {
            return optional_value.value();
        }
        else {
            const auto value = op();
            impl_->add_field(name, value);
            return value;
        }
    }

private:
    mutable dal::detail::pimpl<dataframe_impl> impl_;
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
    explicit dataframe_builder_impl(const std::string& dataset);
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

    explicit dataframe_builder(const std::string& dataset)
            : impl_(new dataframe_builder_impl{ dataset }) {}

    dataframe_builder& fill(double value);

    dataframe_builder& fill_diag(double value);

    dataframe_builder& fill_uniform(double a, double b, std::int64_t seed = 7777);
    dataframe_builder& fill_normal(double mean, double deviation, std::int64_t seed = 7777);

    dataframe build() const;

protected:
    dal::detail::pimpl<dataframe_builder_impl> impl_;
};

} // namespace oneapi::dal::test::engine
