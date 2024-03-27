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

#include "oneapi/dal/algo/basic_statistics/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace bs = oneapi::dal::basic_statistics;

template <typename TestType>
class basic_statistics_spmd_test
        : public basic_statistics_test<TestType, basic_statistics_spmd_test<TestType>> {
public:
    using base_t = basic_statistics_test<TestType, basic_statistics_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using input_t = typename base_t::input_t;
    using result_t = typename base_t::result_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    result_t compute_override(Args &&...args) {
        return this->compute_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    result_t merge_compute_result_override(const std::vector<result_t> &results) {
        return results[0];
    }

    template <typename... Args>
    std::vector<input_t> split_compute_input_override(std::int64_t split_count, Args &&...args) {
        const input_t input{ std::forward<Args>(args)... };

        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);

        std::vector<input_t> split_input;
        split_input.reserve(split_count);

        if (input.get_weights().has_data()) {
            const auto split_weights = te::split_table_by_rows<float_t>(this->get_policy(),
                                                                        input.get_weights(),
                                                                        split_count);

            for (std::int64_t i = 0; i < split_count; i++) {
                split_input.push_back( //
                    input_t{ split_data[i], split_weights[i] });
            }
        }
        else {
            for (std::int64_t i = 0; i < split_count; i++) {
                split_input.push_back( //
                    input_t{ split_data[i] });
            }
        }
        return split_input;
    }

    void spmd_general_checks(const table &data,
                             const table &weights,
                             bs::result_option_id compute_mode) {
        CAPTURE(static_cast<std::uint64_t>(compute_mode));

        const auto bs_desc = base_t::get_descriptor(compute_mode);

        const auto compute_result = this->compute_override(bs_desc, data, weights);

        base_t::check_compute_result(compute_mode, data, weights, compute_result);
        base_t::check_for_exception_for_non_requested_results(compute_mode, compute_result);
    }

private:
    std::int64_t rank_count_;
};

TEMPLATE_LIST_TEST_M(basic_statistics_spmd_test,
                     "basic_statistics common flow",
                     "[basic_statistics][integration][spmd]",
                     basic_statistics_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 530 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 500, 250 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 6000, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 6000, 530 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 10000, 200 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 100, 3000 }.fill_normal(-0.5, 0.5, 7777));

    this->set_rank_count(GENERATE(2, 3));
    std::shared_ptr<te::dataframe> weights;
    const bool use_weights = GENERATE(0, 1);

    if (use_weights) {
        const auto row_count = data.get_row_count();
        weights = std::make_shared<te::dataframe>(
            te::dataframe_builder{ row_count, 1 }.fill_normal(0, 1, 777).build());
    }

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);

    const auto data_table_id = this->get_homogen_table_id();
    const table data_ = data.get_table(this->get_policy(), data_table_id);

    if (use_weights) {
        const auto weights_table_id = this->get_homogen_table_id();
        const table weights_ = weights->get_table(this->get_policy(), weights_table_id);
        this->spmd_general_checks(data_, weights_, compute_mode);
    }
    else {
        this->spmd_general_checks(data_, table{}, compute_mode);
    }
}

} // namespace oneapi::dal::basic_statistics::test
