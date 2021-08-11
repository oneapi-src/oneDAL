/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

using task_t = task::compute;
using descriptor_t = detail::descriptor_base<task_t>;

inline auto get_daal_estimates_to_compute(const descriptor_t& desc) {
    const auto res_op = desc.get_result_options();
    const auto res_min_max = result_options::min | result_options::max;
    const auto res_mean_varc = result_options::mean | result_options::variance;

    if ((res_op.test(res_min_max) && res_op.get_mask() ^ res_min_max.get_mask()) ||
        (res_op.test(res_mean_varc) && res_op.get_mask() ^ res_mean_varc.get_mask())) {
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
inline auto get_result(const daal_lom::Result& daal_result) {
    compute_result<Task> res;

    res.set_min(interop::convert_from_daal_homogen_table<Float>(
        daal_result.get(daal_lom::ResultId::minimum)));
    res.set_max(interop::convert_from_daal_homogen_table<Float>(
        daal_result.get(daal_lom::ResultId::maximum)));
    res.set_sum(
        interop::convert_from_daal_homogen_table<Float>(daal_result.get(daal_lom::ResultId::sum)));
    res.set_sum_squares(interop::convert_from_daal_homogen_table<Float>(
        daal_result.get(daal_lom::ResultId::sumSquares)));
    res.set_sum_squares_centered(interop::convert_from_daal_homogen_table<Float>(
        daal_result.get(daal_lom::ResultId::sumSquaresCentered)));
    res.set_mean(
        interop::convert_from_daal_homogen_table<Float>(daal_result.get(daal_lom::ResultId::mean)));
    res.set_second_order_raw_moment(interop::convert_from_daal_homogen_table<Float>(
        daal_result.get(daal_lom::ResultId::secondOrderRawMoment)));
    res.set_variance(interop::convert_from_daal_homogen_table<Float>(
        daal_result.get(daal_lom::ResultId::variance)));
    res.set_standard_deviation(interop::convert_from_daal_homogen_table<Float>(
        daal_result.get(daal_lom::ResultId::standardDeviation)));
    res.set_variation(interop::convert_from_daal_homogen_table<Float>(
        daal_result.get(daal_lom::ResultId::variation)));

    return res;
}

} // namespace oneapi::dal::basic_statistics::backend
