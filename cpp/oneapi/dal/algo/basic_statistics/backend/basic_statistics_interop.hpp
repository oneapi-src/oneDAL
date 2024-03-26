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

#include "oneapi/dal/algo/basic_statistics/common.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include <daal/include/algorithms/moments/low_order_moments_types.h>

namespace oneapi::dal::basic_statistics::backend {

namespace daal_lom = daal::algorithms::low_order_moments;
namespace interop = dal::backend::interop;
namespace bk = dal::backend;

using task_t = task::compute;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
inline auto get_desc_to_compute(const descriptor_t& desc) {
    const auto res_op = desc.get_result_options();
    bool has_min_max = res_op.test(result_options::min) || res_op.test(result_options::max);
    bool has_other_stat =
        res_op.test(result_options::mean) || res_op.test(result_options::variance) ||
        res_op.test(result_options::second_order_raw_moment) ||
        res_op.test(result_options::variation) || res_op.test(result_options::standard_deviation);
    bool has_sums = res_op.test(result_options::sum) ||
                    res_op.test(result_options::sum_squares_centered) || has_other_stat;
    bool has_sums2 = res_op.test(result_options::sum_squares_centered) ||
                     res_op.test(result_options::sum_squares_centered) || has_other_stat;
    auto local_desc =
        basic_statistics::descriptor<Float, method::dense, basic_statistics::task::compute>();
    if (has_min_max && has_sums && has_sums2) {
        local_desc.set_result_options(result_options::min | result_options::max |
                                      result_options::sum | result_options::sum_squares |
                                      result_options::sum_squares_centered);
    }
    else if (!has_min_max || has_sums || has_sums2) {
        local_desc.set_result_options(result_options::sum | result_options::sum_squares |
                                      result_options::sum_squares_centered);
    }
    else if (has_min_max && !has_sums && !has_sums2) {
        local_desc.set_result_options(result_options::min | result_options::max);
    }
    return local_desc;
}

inline auto get_daal_estimates_to_compute(const descriptor_t& desc) {
    const auto res_op = desc.get_result_options();
    const auto res_min_max = result_options::min | result_options::max;
    const auto res_mean_varc = result_options::mean | result_options::variance;

    if ((res_op.test(res_min_max) && res_op.test(~res_min_max)) ||
        (res_op.test(res_mean_varc) && res_op.test(~res_mean_varc))) {
        return daal_lom::estimatesAll;
    }
    else if (res_op.test(res_min_max)) {
        return daal_lom::estimatesMinMax;
    }
    else if (res_op.test(res_mean_varc)) {
        return daal_lom::estimatesMeanVariance;
    }

    return daal_lom::estimatesAll;
}

template <typename Float, typename Task>
inline auto get_result(const descriptor_t& desc, const daal_lom::Result& daal_result) {
    compute_result<Task> res;

    const auto res_op = desc.get_result_options();
    res.set_result_options(desc.get_result_options());

    if (res_op.test(result_options::min)) {
        res.set_min(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::minimum)));
    }
    if (res_op.test(result_options::max)) {
        res.set_max(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::maximum)));
    }
    if (res_op.test(result_options::sum)) {
        res.set_sum(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::sum)));
    }
    if (res_op.test(result_options::sum_squares)) {
        res.set_sum_squares(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::sumSquares)));
    }
    if (res_op.test(result_options::sum_squares_centered)) {
        res.set_sum_squares_centered(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::sumSquaresCentered)));
    }
    if (res_op.test(result_options::mean)) {
        res.set_mean(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::mean)));
    }
    if (res_op.test(result_options::second_order_raw_moment)) {
        res.set_second_order_raw_moment(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::secondOrderRawMoment)));
    }
    if (res_op.test(result_options::variance)) {
        res.set_variance(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::variance)));
    }
    if (res_op.test(result_options::standard_deviation)) {
        res.set_standard_deviation(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::standardDeviation)));
    }
    if (res_op.test(result_options::variation)) {
        res.set_variation(interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(daal_lom::ResultId::variation)));
    }

    return res;
}

template <typename Float>
inline array<Float> copy_immutable(const array<Float>&& inp) {
    if (inp.has_mutable_data()) {
        return inp;
    }
    else {
        const auto count = inp.get_count();
        auto res = array<Float>::empty(count);
        bk::copy(res.get_mutable_data(), inp.get_data(), count);
        return res;
    }
}

template <typename Float, typename Result, typename Input, typename Parameter>
inline void alloc_result(Result& result, const Input* input, const Parameter* params, int method) {
    const auto status = result.template allocate<Float>(input, params, method);
    interop::status_to_exception(status);
}

template <typename Float, typename Result, typename Input, typename Parameter>
inline void initialize_result(Result& result,
                              const Input* input,
                              const Parameter* params,
                              int method) {
    const auto status = result.template initialize<Float>(input, params, method);
    interop::status_to_exception(status);
}

} // namespace oneapi::dal::basic_statistics::backend
