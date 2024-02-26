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

#include "oneapi/dal/algo/covariance/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::covariance::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace cov = oneapi::dal::covariance;

template <typename TestType>
class covariance_online_spmd_test
        : public covariance_test<TestType, covariance_online_spmd_test<TestType>> {
public:
    using base_t = covariance_test<TestType, covariance_online_spmd_test<TestType>>;
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

    std::vector<result_t> merge_finalize_compute_result_override(
        const std::vector<result_t>& results) {
        return results;
    }

    template <typename... Args>
    std::vector<partial_result_t> split_finalize_compute_input_override(std::int64_t split_count,
                                                                        Args&&... args) {
        ONEDAL_ASSERT(split_count == rank_count_);
        const std::vector<partial_result_t> input{ std::forward<Args>(args)... };

        return input;
    }

    void online_spmd_general_checks(const te::dataframe& data_fr,
                                    cov::result_option_id compute_mode,
                                    const te::table_id& data_table_id) {
        CAPTURE(static_cast<std::uint64_t>(compute_mode));
        const table data = data_fr.get_table(this->get_policy(), data_table_id);

        const auto cov_desc = base_t::get_descriptor(compute_mode);
        std::vector<partial_result_t> partial_results;
        auto input_table = base_t::template split_table_by_rows<double>(data, rank_count_);
        for (int64_t i = 0; i < rank_count_; i++) {
            dal::covariance::partial_compute_result<> partial_result;
            auto input_table_blocks =
                base_t::template split_table_by_rows<double>(input_table[i], blocks_count_);
            for (int64_t j = 0; j < blocks_count_; j++) {
                partial_result =
                    this->partial_compute(cov_desc, partial_result, input_table_blocks[j]);
            }
            partial_results.push_back(partial_result);
        }
        const auto compute_result = this->finalize_compute_override(cov_desc, partial_results);
        for (int64_t i = 0; i < rank_count_; i++) {
            base_t::check_compute_result(cov_desc, data, compute_result[i]);
        }
    }

private:
    std::int64_t rank_count_;
    std::int64_t blocks_count_;
};

using covariance_types = COMBINE_TYPES((float, double), (covariance::method::dense));

TEMPLATE_LIST_TEST_M(covariance_online_spmd_test,
                     "covariance common flow",
                     "[covariance][integration][spmd]",
                     covariance_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1000, 100 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 2000, 20 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 2500, 20 }.fill_normal(-30, 30, 7777));
    this->set_rank_count(GENERATE(1, 2, 4));
    this->set_blocks_count(GENERATE(1, 3, 10));
    cov::result_option_id mode_mean = result_options::means;
    cov::result_option_id mode_cov = result_options::cov_matrix;
    cov::result_option_id mode_cor = result_options::cor_matrix;
    cov::result_option_id mode_cov_mean = result_options::cov_matrix | result_options::means;
    cov::result_option_id mode_cov_cor = result_options::cov_matrix | result_options::cor_matrix;
    cov::result_option_id mode_cor_mean = result_options::cor_matrix | result_options::means;
    cov::result_option_id res_all =
        result_options::cov_matrix | result_options::cor_matrix | result_options::means;

    const cov::result_option_id compute_mode = GENERATE_COPY(mode_mean,
                                                             mode_cor,
                                                             mode_cov,
                                                             mode_cor_mean,
                                                             mode_cov_mean,
                                                             mode_cov_cor,
                                                             res_all);

    const auto data_table_id = this->get_homogen_table_id();

    this->online_spmd_general_checks(data, compute_mode, data_table_id);
}

} // namespace oneapi::dal::covariance::test
