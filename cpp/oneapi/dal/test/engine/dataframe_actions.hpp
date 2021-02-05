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

#include <random>

#include "oneapi/dal/test/engine/dataframe_common.hpp"

namespace oneapi::dal::test::engine {

class dataframe_builder_action_allocate : public dataframe_builder_action {
public:
    explicit dataframe_builder_action_allocate(std::int64_t row_count, std::int64_t column_count)
            : row_count_(row_count),
              column_count_(column_count) {
        if (row_count == 0 || column_count == 0) {
            throw invalid_argument{ fmt::format(
                "Invalid dataframe shape, row and column count must be positive, "
                "but got row_count = {}, column_count = {}",
                row_count,
                column_count) };
        }
    }

    std::string get_opcode() const override {
        return fmt::format("allocate({},{})", row_count_, column_count_);
    }

    dataframe_impl* execute(dataframe_impl* df) const override {
        delete df;
        const auto arr = array<float>::empty(row_count_ * column_count_);
        return new dataframe_impl{ arr, row_count_, column_count_ };
    }

private:
    std::int64_t row_count_;
    std::int64_t column_count_;
};

class dataframe_builder_action_fill_uniform : public dataframe_builder_action {
public:
    explicit dataframe_builder_action_fill_uniform(double a, double b, std::int64_t seed)
            : a_(a),
              b_(b),
              seed_(seed) {
        if (a >= b) {
            throw invalid_argument{ fmt::format("Invalid uniform distribution interval, "
                                                "expected b > a, but got a = {}, b = {}",
                                                a,
                                                b) };
        }
    }

    std::string get_opcode() const override {
        return fmt::format("fill_uniform({},{},{})", a_, b_, seed_);
    }

    dataframe_impl* execute(dataframe_impl* df) const override {
        if (!df) {
            throw invalid_argument{ "Action fill_uniform got null dataframe" };
        }

        float* data = df->get_array().need_mutable_data().get_mutable_data();

        // TODO: Migrate to MKL's random generators
        std::mt19937 rng(seed_);

        std::uniform_real_distribution<float> distr(a_, b_);
        for (std::int64_t i = 0; i < df->get_count(); i++) {
            data[i] = distr(rng);
        }

        return df;
    }

private:
    double a_ = 0.0;
    double b_ = 1.0;
    std::int64_t seed_ = 7777;
};

class dataframe_builder_action_fill : public dataframe_builder_action {
public:
    explicit dataframe_builder_action_fill(double value) : value_(value) {}

    std::string get_opcode() const override {
        return fmt::format("fill({})", value_);
    }

    dataframe_impl* execute(dataframe_impl* df) const override {
        if (!df) {
            throw invalid_argument{ "Action fill got null dataframe" };
        }

        float* data = df->get_array().need_mutable_data().get_mutable_data();
        for (std::int64_t i = 0; i < df->get_count(); i++) {
            data[i] = value_;
        }

        return df;
    }

private:
    double value_ = 0.0;
};

class dataframe_builder_action_fill_diag : public dataframe_builder_action {
public:
    explicit dataframe_builder_action_fill_diag(double value) : value_(value) {}

    std::string get_opcode() const override {
        return fmt::format("fill_diag({})", value_);
    }

    dataframe_impl* execute(dataframe_impl* df) const override {
        if (!df) {
            throw invalid_argument{ "Action fill_diag got null dataframe" };
        }

        const std::int64_t column_count = df->get_column_count();
        float* data = df->get_array().need_mutable_data().get_mutable_data();

        for (std::int64_t i = 0; i < df->get_count(); i++) {
            data[i] = 0;
        }

        for (std::int64_t i = 0; i < column_count; i++) {
            data[i * column_count + i] = value_;
        }

        return df;
    }

private:
    double value_ = 0.0;
};

} // namespace oneapi::dal::test::engine
