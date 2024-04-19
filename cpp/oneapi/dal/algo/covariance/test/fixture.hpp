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

#include "oneapi/dal/algo/covariance/compute.hpp"
#include "oneapi/dal/algo/covariance/partial_compute.hpp"
#include "oneapi/dal/algo/covariance/finalize_compute.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::covariance::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace cov = oneapi::dal::covariance;

template <typename TestType, typename Derived>
class covariance_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using input_t = cov::compute_input<>;
    using partial_input_t = cov::partial_compute_input<>;
    using partial_result_t = cov::partial_compute_result<>;
    using result_t = cov::compute_result<>;
    using descriptor_t = cov::descriptor<Float, Method>;

    auto get_descriptor(cov::result_option_id compute_mode) const {
        return descriptor_t{}.set_result_options(compute_mode);
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    template <typename Float>
    std::vector<dal::table> split_table_by_rows(const dal::table& t, std::int64_t split_count) {
        ONEDAL_ASSERT(0l < split_count);
        ONEDAL_ASSERT(split_count <= t.get_row_count());

        const std::int64_t row_count = t.get_row_count();
        const std::int64_t column_count = t.get_column_count();
        const std::int64_t block_size_regular = row_count / split_count;
        const std::int64_t block_size_tail = row_count % split_count;

        std::vector<dal::table> result(split_count);

        std::int64_t row_offset = 0;
        for (std::int64_t i = 0; i < split_count; i++) {
            const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
            const std::int64_t block_size = block_size_regular + tail;

            const auto row_range = dal::range{ row_offset, row_offset + block_size };
            const auto block = dal::row_accessor<const Float>{ t }.pull(row_range);
            result[i] = dal::homogen_table::wrap(block, block_size, column_count);
            row_offset += block_size;
        }

        return result;
    }
    void general_checks_assume_centered(const te::dataframe& input,
                                        const te::table_id& input_table_id) {
        const table data = input.get_table(this->get_policy(), input_table_id);

        INFO("create descriptor means assume centered");
        auto cov_desc = covariance::descriptor<Float, Method, covariance::task::compute>()
                            .set_result_options(covariance::result_options::means)
                            .set_assume_centered(true);
        INFO("run compute optional: means assume centered");
        auto compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov assume centered");
        cov_desc = covariance::descriptor<Float, Method, covariance::task::compute>()
                       .set_result_options(covariance::result_options::cov_matrix)
                       .set_assume_centered(true);
        INFO("run compute optional: cov assume centered");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov means");
        cov_desc = covariance::descriptor<Float, Method, covariance::task::compute>()
                       .set_result_options(covariance::result_options::cov_matrix |
                                           covariance::result_options::means)
                       .set_assume_centered(true);
        INFO("run compute optional: cov means");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov cor means");
        cov_desc = covariance::descriptor<Float, Method, covariance::task::compute>()
                       .set_result_options(covariance::result_options::cov_matrix |
                                           covariance::result_options::cor_matrix |
                                           covariance::result_options::means)
                       .set_assume_centered(true);
        INFO("run compute optional: cov cor means");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);
    }
    void general_checks(const te::dataframe& input, const te::table_id& input_table_id) {
        const table data = input.get_table(this->get_policy(), input_table_id);

        INFO("create descriptor cov cor means");
        auto cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::cor_matrix |
                covariance::result_options::means);
        INFO("run compute optional: cov cor means");
        auto compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix);
        INFO("run compute optional: cov");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov biased");
        cov_desc = covariance::descriptor<Float, Method, covariance::task::compute>()
                       .set_result_options(covariance::result_options::cov_matrix)
                       .set_bias(true);
        INFO("run compute optional: cov biased");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cor");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cor_matrix);
        INFO("run compute optional: cor");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor means");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::means);
        INFO("run compute optional: means");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov cor");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::cor_matrix);
        INFO("run compute optional: cov cor");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov means");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::means);
        INFO("run compute optional: cov means");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cor means");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cor_matrix | covariance::result_options::means);
        INFO("run compute optional: cor means");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(cov_desc, data, compute_result);
    }

    void online_general_checks(const te::dataframe& input,
                               const te::table_id& input_table_id,
                               const std::int64_t nBlocks) {
        const table data = input.get_table(this->get_policy(), input_table_id);
        INFO("create descriptor cov cor means");
        auto cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::cor_matrix |
                covariance::result_options::means);
        INFO("run compute optional: cov cor means");
        dal::covariance::partial_compute_result<> partial_result;
        auto input_table = split_table_by_rows<double>(data, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; ++i) {
            partial_result = this->partial_compute(cov_desc, partial_result, input_table[i]);
        }
        auto compute_result = this->finalize_compute(cov_desc, partial_result);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix);
        INFO("run compute optional: cov");
        partial_result = dal::covariance::partial_compute_result();
        input_table = split_table_by_rows<double>(data, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; ++i) {
            partial_result = this->partial_compute(cov_desc, partial_result, input_table[i]);
        }
        compute_result = this->finalize_compute(cov_desc, partial_result);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cor");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cor_matrix);
        INFO("run compute optional: cor");
        partial_result = dal::covariance::partial_compute_result();
        input_table = split_table_by_rows<double>(data, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; ++i) {
            partial_result = this->partial_compute(cov_desc, partial_result, input_table[i]);
        }
        compute_result = this->finalize_compute(cov_desc, partial_result);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor means");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::means);
        INFO("run compute optional: means");
        partial_result = dal::covariance::partial_compute_result();
        input_table = split_table_by_rows<double>(data, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; ++i) {
            partial_result = this->partial_compute(cov_desc, partial_result, input_table[i]);
        }
        compute_result = this->finalize_compute(cov_desc, partial_result);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov cor");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::cor_matrix);
        INFO("run compute optional: cov cor");
        partial_result = dal::covariance::partial_compute_result();
        input_table = split_table_by_rows<double>(data, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; ++i) {
            partial_result = this->partial_compute(cov_desc, partial_result, input_table[i]);
        }
        compute_result = this->finalize_compute(cov_desc, partial_result);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cov means");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::means);
        INFO("run compute optional: cov means");
        partial_result = dal::covariance::partial_compute_result();
        input_table = split_table_by_rows<double>(data, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; ++i) {
            partial_result = this->partial_compute(cov_desc, partial_result, input_table[i]);
        }
        compute_result = this->finalize_compute(cov_desc, partial_result);
        check_compute_result(cov_desc, data, compute_result);

        INFO("create descriptor cor means");
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cor_matrix | covariance::result_options::means);
        INFO("run compute optional: cor means");
        partial_result = dal::covariance::partial_compute_result();
        input_table = split_table_by_rows<double>(data, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; ++i) {
            partial_result = this->partial_compute(cov_desc, partial_result, input_table[i]);
        }
        compute_result = this->finalize_compute(cov_desc, partial_result);
        check_compute_result(cov_desc, data, compute_result);
    }

    void check_compute_result(const covariance::descriptor<Float, Method>& desc,
                              const table& data,
                              const covariance::compute_result<>& result) {
        if (result.get_result_options().test(result_options::cov_matrix)) {
            const auto cov_matrix = result.get_cov_matrix();
            INFO("check if cov matrix table shape is expected");
            REQUIRE(cov_matrix.get_row_count() == data.get_column_count());
            REQUIRE(cov_matrix.get_column_count() == data.get_column_count());
            INFO("check if there is no NaN in cov matrix table");
            REQUIRE(te::has_no_nans(cov_matrix));
            INFO("check if cov matrix values are expected");
            check_cov_matrix_values(desc, data, cov_matrix);
        }
        if (result.get_result_options().test(result_options::cor_matrix)) {
            const auto cor_matrix = result.get_cor_matrix();
            INFO("check if cor matrix table shape is expected");
            REQUIRE(cor_matrix.get_row_count() == data.get_column_count());
            REQUIRE(cor_matrix.get_column_count() == data.get_column_count());
            INFO("check if there is no NaN in cor matrix table");
            REQUIRE(te::has_no_nans(cor_matrix));
            INFO("check if cor matrix values are expected");
            check_cor_matrix_values(desc, data, cor_matrix);
        }
        if (result.get_result_options().test(result_options::means)) {
            const auto means = result.get_means();
            INFO("check if means table shape is expected");
            REQUIRE(means.get_row_count() == 1);
            REQUIRE(means.get_column_count() == data.get_column_count());
            INFO("check if there is no NaN in means table");
            REQUIRE(te::has_no_nans(means));
            INFO("check if means values are expected");
            check_means_values(desc, data, means);
        }
    }

    void check_means_values(const covariance::descriptor<Float, Method>& desc,
                            const table& data,
                            const table& means) {
        const auto reference_means = compute_reference_means(data, desc.get_assume_centered());
        const double tol = te::get_tolerance<Float>(1e-4, 1e-9);
        const double diff = te::abs_error(reference_means, means);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_means(const table& data, bool assume_centered = false) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto row_count_data = data_matrix.get_row_count();
        const auto column_count_data = data_matrix.get_column_count();
        auto reference_means = la::matrix<double>::full({ 1, column_count_data }, 0.0);
        if (!assume_centered) {
            for (std::int64_t i = 0; i < column_count_data; i++) {
                double sum = 0;
                for (std::int64_t j = 0; j < row_count_data; j++) {
                    sum += data_matrix.get(j, i);
                }
                reference_means.set(0, i) = sum / row_count_data;
            }
        }

        return reference_means;
    }

    void check_cov_matrix_values(const covariance::descriptor<Float, Method>& desc,
                                 const table& data,
                                 const table& cov_matrix) {
        const auto reference_cov = compute_reference_cov(desc, data);
        const auto data_matrix = la::matrix<double>::wrap(cov_matrix);
        const double tol = te::get_tolerance<Float>(1e-2, 1e-9);
        const double diff = te::abs_error(reference_cov, cov_matrix);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_cov(const covariance::descriptor<Float, Method>& desc,
                                             const table& data) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto row_count_data = data_matrix.get_row_count();
        const auto column_count_data = data_matrix.get_column_count();
        auto reference_means = compute_reference_means(data, desc.get_assume_centered());
        auto reference_cov =
            la::matrix<double>::full({ column_count_data, column_count_data }, 0.0);
        auto multiplier = 1 / static_cast<double>(row_count_data - 1);
        if (desc.get_bias()) {
            multiplier = 1 / static_cast<double>(row_count_data);
        }
        for (std::int64_t i = 0; i < column_count_data; i++) {
            for (std::int64_t j = 0; j < column_count_data; j++) {
                double elem = 0;
                for (std::int64_t k = 0; k < row_count_data; k++) {
                    elem += (data_matrix.get(k, i) - reference_means.get(0, i)) *
                            (data_matrix.get(k, j) - reference_means.get(0, j));
                }
                reference_cov.set(i, j) = elem * multiplier;
            }
        }
        return reference_cov;
    }
    void check_cor_matrix_values(const covariance::descriptor<Float, Method>& desc,
                                 const table& data,
                                 const table& cor_matrix) {
        const auto reference_cor = compute_reference_cor(desc, data);
        const double tol = te::get_tolerance<Float>(1e-2, 1e-9);
        const double diff = te::abs_error(reference_cor, cor_matrix);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_cor(const covariance::descriptor<Float, Method>& desc,
                                             const table& data) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto column_count_data = data_matrix.get_column_count();
        auto reference_means = compute_reference_means(data, desc.get_assume_centered());
        auto reference_cov = compute_reference_cov(desc, data);
        auto reference_cor =
            la::matrix<double>::full({ column_count_data, column_count_data }, 0.0);
        for (std::int64_t i = 0; i < column_count_data; i++) {
            for (std::int64_t j = 0; j < column_count_data; j++) {
                reference_cor.set(i, j) =
                    reference_cov.get(i, j) /
                    std::sqrt(reference_cov.get(i, i) * reference_cov.get(j, j));
            }
        }

        return reference_cor;
    }
};

using covariance_types = COMBINE_TYPES((float, double), (covariance::method::dense));

} // namespace oneapi::dal::covariance::test
