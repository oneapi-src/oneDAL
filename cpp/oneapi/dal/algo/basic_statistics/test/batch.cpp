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

#include "oneapi/dal/algo/basic_statistics/compute.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace bs = oneapi::dal::basic_statistics;

constexpr inline std::uint64_t mask_full = 0xffffffffffffffff;

template <typename TestType>
class basic_statistics_batch_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using result_t = bs::compute_result<>;
    using descriptor_t = bs::descriptor<Float, Method>;

    auto get_descriptor(bs::result_option_id compute_mode) const {
        return descriptor_t{}.set_result_options(compute_mode);
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    void general_checks(const te::dataframe& data_fr,
                        bs::result_option_id compute_mode,
                        const te::table_id& data_table_id) {
        CAPTURE(compute_mode);
        const table data = data_fr.get_table(this->get_policy(), data_table_id);

        const auto bs_desc = get_descriptor(compute_mode);

        const auto compute_result = this->compute(bs_desc, data);

        check_compute_result(compute_mode, data, compute_result);
    }

    void check_compute_result(bs::result_option_id compute_mode,
                              const table& data,
                              const result_t& result) {
        SECTION("result tables' shape is expected") {
            check_result_shape(compute_mode, data, result);
        }

        SECTION("check results against reference") {
            check_vs_reference(compute_mode, data, result);
        }
    }

    void check_result_shape(bs::result_option_id compute_mode,
                            const table& data,
                            const result_t& result) {
        if (compute_mode.test(res_min_max)) {
            REQUIRE(result.get_min().get_column_count() == data.get_column_count());
            REQUIRE(result.get_max().get_column_count() == data.get_column_count());
        }

        if (compute_mode.test(res_mean_varc)) {
            REQUIRE(result.get_mean().get_column_count() == data.get_column_count());
            REQUIRE(result.get_variance().get_column_count() == data.get_column_count());
        }

        if ((compute_mode.test(res_min_max) && compute_mode.get_mask() ^ res_min_max.get_mask()) ||
            (compute_mode.test(res_mean_varc) &&
             compute_mode.get_mask() ^ res_mean_varc.get_mask())) {
            REQUIRE(result.get_sum().get_column_count() == data.get_column_count());
            REQUIRE(result.get_sum_squares().get_column_count() == data.get_column_count());
            REQUIRE(result.get_sum_squares_centered().get_column_count() ==
                    data.get_column_count());
            REQUIRE(result.get_second_order_raw_moment().get_column_count() ==
                    data.get_column_count());
            REQUIRE(result.get_standard_deviation().get_column_count() == data.get_column_count());
            REQUIRE(result.get_variation().get_column_count() == data.get_column_count());
        }
    }

    void check_vs_reference(bs::result_option_id compute_mode,
                            const table& data,
                            const result_t& result) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto row_count = data_matrix.get_row_count();
        const auto column_count = data_matrix.get_column_count();
        auto ref_min = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_max = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_sum = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_sum2 = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_sum2cent = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_mean = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_sorm = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_varc = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_stdev = la::matrix<double>::full({ 1, column_count }, 0.0);
        auto ref_vart = la::matrix<double>::full({ 1, column_count }, 0.0);

        //init min max
        for (std::int64_t clmn = 0; clmn < column_count; clmn++) {
            ref_min.set(0, clmn) = data_matrix.get(0, clmn);
            ref_max.set(0, clmn) = data_matrix.get(0, clmn);
        }

        // calc mean
        for (std::int64_t row = 0; row < row_count; row++) {
            for (std::int64_t clmn = 0; clmn < column_count; clmn++) {
                ref_mean.set(0, clmn) += data_matrix.get(row, clmn);
            }
        }

        // finalize mean
        for (std::int64_t clmn = 0; clmn < column_count; clmn++) {
            ref_mean.set(0, clmn) = ref_mean.set(0, clmn) / Float(row_count);
        }

        for (std::int64_t row = 0; row < row_count; row++) {
            for (std::int64_t clmn = 0; clmn < column_count; clmn++) {
                ref_min.set(0, clmn) = std::min(ref_min.get(0, clmn), data_matrix.get(row, clmn));
                ref_max.set(0, clmn) = std::max(ref_max.get(0, clmn), data_matrix.get(row, clmn));
                ref_sum.set(0, clmn) += data_matrix.get(row, clmn);
                ref_sum2.set(0, clmn) += data_matrix.get(row, clmn) * data_matrix.get(row, clmn);
                ref_sum2cent.set(0, clmn) += (data_matrix.get(row, clmn) - ref_mean.get(0, clmn)) *
                                             (data_matrix.get(row, clmn) - ref_mean.get(0, clmn));
                ref_sorm.set(0, clmn) = ref_sum2.get(0, clmn) / Float(row);
                ref_varc.set(0, clmn) = ref_sum2cent.get(0, clmn) / Float(row - 1);
                ref_stdev.set(0, clmn) = std::sqrt(ref_varc.get(0, clmn));
                ref_vart.set(0, clmn) = ref_stdev.get(0, clmn) / ref_mean.get(0, clmn);
            }
        }

        if (compute_mode.test(res_min_max)) {
            check_arr_vs_ref(ref_min, la::matrix<double>::wrap(result.get_min()), "Min");
            check_arr_vs_ref(ref_max, la::matrix<double>::wrap(result.get_max()), "Max");
        }

        if (compute_mode.test(res_mean_varc)) {
            check_arr_vs_ref(ref_mean, la::matrix<double>::wrap(result.get_mean()), "Mean");
            check_arr_vs_ref(ref_varc, la::matrix<double>::wrap(result.get_variance()), "Varc");
        }

        if ((compute_mode.test(res_min_max) && compute_mode.get_mask() ^ res_min_max.get_mask()) ||
            (compute_mode.test(res_mean_varc) &&
             compute_mode.get_mask() ^ res_mean_varc.get_mask())) {
            check_arr_vs_ref(ref_sum, la::matrix<double>::wrap(result.get_sum()), "Sum");
            check_arr_vs_ref(ref_sum2, la::matrix<double>::wrap(result.get_sum_squares()), "Sum2");
            check_arr_vs_ref(ref_sum2cent,
                             la::matrix<double>::wrap(result.get_sum_squares_centered()),
                             "Sum2Cent");
            check_arr_vs_ref(ref_sorm,
                             la::matrix<double>::wrap(result.get_second_order_raw_moment()),
                             "SORM");
            check_arr_vs_ref(ref_stdev,
                             la::matrix<double>::wrap(result.get_standard_deviation()),
                             "StDev");
            check_arr_vs_ref(ref_vart, la::matrix<double>::wrap(result.get_variation()), "Vart");
        }
    }

    void check_arr_vs_ref(const la::matrix<double>& ref,
                          const la::matrix<double>& res,
                          std::string info = "") {
        CAPTURE(info);
        const double tol = 1e-1;
        const double diff = te::rel_error(ref, res, 0.0);
        CHECK(diff < tol);
    }

private:
    bs::result_option_id res_min_max = result_options::min | result_options::max;
    bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    bs::result_option_id res_all = bs::result_option_id(dal::result_option_id_base(mask_full));
};

using basic_statistics_types = COMBINE_TYPES((float, double), (basic_statistics::method::dense));

TEMPLATE_LIST_TEST_M(basic_statistics_batch_test,
                     "basic_statistics common flow",
                     "[basic_statistics][integration][batch]",
                     basic_statistics_types) {
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 530 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 500, 250 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 6000, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 6000, 530 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 10000, 200 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 1000000, 20 }.fill_normal(-0.5, 0.5, 7777));

    bs::result_option_id res_min_max = result_options::min | result_options::max;
    bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    bs::result_option_id res_all = bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);

    const auto data_table_id = this->get_homogen_table_id();

    this->general_checks(data, compute_mode, data_table_id);
}

} // namespace oneapi::dal::basic_statistics::test
