/*******************************************************************************
* Copyright contributors to the oneDAL project
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
class basic_statistics_online_spmd_test
        : public basic_statistics_test<TestType, basic_statistics_online_spmd_test<TestType>> {
public:
    using base_t = basic_statistics_test<TestType, basic_statistics_online_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using input_t = typename base_t::input_t;
    using partial_input_t = typename base_t::partial_input_t;
    using partial_result_t = typename base_t::partial_result_t;
    using result_t = typename base_t::result_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    void set_blocks_count(std::int64_t blocks_count) {
        blocks_count_ = blocks_count;
    }

    template <typename... Args>
    result_t finalize_compute_override(Args&&... args) {
        return this->finalize_compute_via_spmd_threads_and_merge(rank_count_,
                                                                 std::forward<Args>(args)...);
    }

    result_t merge_finalize_compute_result_override(const std::vector<result_t>& results) {
        return results[0];
    }

    template <typename... Args>
    std::vector<partial_result_t> split_finalize_compute_input_override(std::int64_t split_count,
                                                                        Args&&... args) {
        ONEDAL_ASSERT(split_count == rank_count_);
        const std::vector<partial_result_t> input{ std::forward<Args>(args)... };

        return input;
    }

    void online_spmd_general_checks(const te::dataframe& data_fr,
                                    bs::result_option_id compute_mode,
                                    const te::table_id& data_table_id) {
        CAPTURE(static_cast<std::uint64_t>(compute_mode));
        const table data = data_fr.get_table(this->get_policy(), data_table_id);

        const auto bs_desc = base_t::get_descriptor(compute_mode);
        std::vector<partial_result_t> partial_results;
        auto input_table = base_t::template split_table_by_rows<double>(data, rank_count_);
        for (int64_t i = 0; i < rank_count_; i++) {
            dal::basic_statistics::partial_compute_result<> partial_result;
            auto input_table_blocks =
                base_t::template split_table_by_rows<double>(input_table[i], blocks_count_);
            for (int64_t j = 0; j < blocks_count_; j++) {
                partial_result =
                    this->partial_compute(bs_desc, partial_result, input_table_blocks[j]);
            }
            partial_results.push_back(partial_result);
        }
        const auto compute_result = this->finalize_compute_override(bs_desc, partial_results);

        base_t::check_compute_result(compute_mode, data, table{}, compute_result);
    }

private:
    std::int64_t rank_count_;
    std::int64_t blocks_count_;
};

using basic_statistics_types = COMBINE_TYPES((float, double), (basic_statistics::method::dense));

TEMPLATE_LIST_TEST_M(basic_statistics_online_spmd_test,
                     "basic_statistics common flow",
                     "[basic_statistics][integration][spmd]",
                     basic_statistics_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1000, 100 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 2000, 20 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 2500, 20 }.fill_normal(-30, 30, 7777));
    this->set_rank_count(GENERATE(1, 2, 4));
    this->set_blocks_count(GENERATE(1, 3, 10));

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);
    const auto data_table_id = this->get_homogen_table_id();
    this->online_spmd_general_checks(data, compute_mode, data_table_id);
}

} // namespace oneapi::dal::basic_statistics::test
