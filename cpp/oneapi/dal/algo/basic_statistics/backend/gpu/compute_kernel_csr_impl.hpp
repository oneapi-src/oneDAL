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

enum stat { min, max, sum, sum_sq, sum_sq_cent, mean, moment2, variance, stddev, variation };

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

    result_t get_result(sycl::queue q,
                        const pr::ndarray<Float, 2> computed_result,
                        result_option_id requested_results) {
        result_t res;
        res.set_result_options(requested_results);
        if (requested_results.test(result_options::min)) {
            auto index = get_result_option_index(result_options::min);
            res.set_min(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::max)) {
            auto index = get_result_option_index(result_options::max);
            res.set_max(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::sum)) {
            auto index = get_result_option_index(result_options::sum);
            res.set_sum(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::sum_squares)) {
            auto index = get_result_option_index(result_options::sum_squares);
            res.set_sum_squares(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::sum_squares_centered)) {
            auto index = get_result_option_index(result_options::sum_squares_centered);
            res.set_sum_squares_centered(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::mean)) {
            auto index = get_result_option_index(result_options::mean);
            res.set_mean(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::second_order_raw_moment)) {
            auto index = get_result_option_index(result_options::second_order_raw_moment);
            res.set_second_order_raw_moment(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::variance)) {
            auto index = get_result_option_index(result_options::variance);
            res.set_variance(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::standard_deviation)) {
            auto index = get_result_option_index(result_options::standard_deviation);
            res.set_standard_deviation(get_result_table(q, computed_result, index));
        }
        if (requested_results.test(result_options::variation)) {
            auto index = get_result_option_index(result_options::variation);
            res.set_variation(get_result_table(q, computed_result, index));
        }
        return res;
    }

    table get_result_table(sycl::queue q,
                           const pr::ndarray<Float, 2> computed_result,
                           std::int32_t index) {
        auto column_count = computed_result.get_shape()[1];
        const auto arr = dal::array<Float>::empty(column_count);
        dal::backend::copy_usm2host(q,
                                    arr.get_mutable_data(),
                                    computed_result.get_data() + index * column_count,
                                    column_count)
            .wait_and_throw();
        return homogen_table::wrap(arr, 1, column_count);
    }

    std::int32_t get_result_option_index(result_option_id opt) {
        std::int32_t index = 0;
        while (!opt.test(res_options_[index]))
            ++index;
        return index;
    }
};

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL