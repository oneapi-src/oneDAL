/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

enum stat {
    min,
    max,
    sum,
    sum_sq,
    sum_sq_cent,
    mean,
    moment2,
    variance,
    stddev,
    variation
};

template <typename Float>
class compute_kernel_csr_impl {
    using method_t = method::sparse;
    using task_t = task::compute;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using input_t = compute_input<task_t, dal::csr_table>;
    using result_t = compute_result<task_t>;
    using descriptor_t = detail::descriptor_base<task_t>;

public:
    result_t operator()(const bk::context_gpu& ctx, const descriptor_t& desc, const input_t& input);

private:
    // Number of different basic statistics
    static const std::int32_t res_opt_count_ = 10;
    // An array of basic statistics
    const result_option_id res_options_[res_opt_count_] = { result_options::min,
                                                          result_options::max,
                                                          result_options::sum,
                                                          result_options::sum_squares,
                                                          result_options::sum_squares_centered,
                                                          result_options::mean,
                                                          result_options::second_order_raw_moment,
                                                          result_options::variance,
                                                          result_options::standard_deviation,
                                                          result_options::variation };

    result_t get_result(sycl::queue q, const pr::ndarray<Float, 2> computed_result, result_option_id requested_results) {
        result_t res;
        auto data_host = computed_result.to_host(q).get_data();
        auto column_count = computed_result.get_shape()[1];
        res.set_result_options(requested_results);
        if (requested_results.test(result_options::min)) {
            auto index = get_result_option_index(result_options::min);
            res.set_min(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::max)) {
            auto index = get_result_option_index(result_options::max);
            res.set_max(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::sum)) {
            auto index = get_result_option_index(result_options::sum);
            res.set_sum(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::sum_squares)) {
            auto index = get_result_option_index(result_options::sum_squares);
            res.set_sum_squares(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::sum_squares_centered)) {
            auto index = get_result_option_index(result_options::sum_squares_centered);
            res.set_sum_squares_centered(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::mean)) {
            auto index = get_result_option_index(result_options::mean);
            res.set_mean(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::second_order_raw_moment)) {
            auto index = get_result_option_index(result_options::second_order_raw_moment);
            res.set_second_order_raw_moment(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::variance)) {
            auto index = get_result_option_index(result_options::variance);
            res.set_variance(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::standard_deviation)) {
            auto index = get_result_option_index(result_options::standard_deviation);
            res.set_standard_deviation(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        if (requested_results.test(result_options::variation)) {
            auto index = get_result_option_index(result_options::variation);
            res.set_variation(homogen_table::wrap(data_host + index * column_count, 1, column_count));
        }
        return res;
    }

    std::int32_t get_result_option_index(result_option_id opt) {
        std::int32_t index = 0;
        while (!opt.test(res_options_[index])) ++index;
        return index;
    }
};

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL