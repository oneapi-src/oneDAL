/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gtest/gtest.h"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;

template <typename T>
inline double calculate_classification_error(const dal::table& infer_labels,
                                             const T* ground_truth) {
    const auto labels = dal::row_accessor<const T>(infer_labels).pull();
    std::int64_t incorrect_label_count = 0;

    for (std::int64_t i = 0; i < labels.get_count(); i++) {
        incorrect_label_count += (static_cast<int>(labels[i]) != static_cast<int>(ground_truth[i]));
    }
    return static_cast<double>(incorrect_label_count) / labels.get_count();
}

template <typename T>
inline double calculate_classification_error(const dal::table& infer_labels,
                                             const dal::homogen_table& ground_truth) {
    const auto labels = dal::row_accessor<const T>(infer_labels).pull();
    const auto truth_labels = dal::row_accessor<const T>(ground_truth).pull();
    std::int64_t incorrect_label_count = 0;

    for (std::int64_t i = 0; i < labels.get_count(); i++) {
        incorrect_label_count += (static_cast<int>(labels[i]) != static_cast<int>(truth_labels[i]));
    }
    return static_cast<double>(incorrect_label_count) / labels.get_count();
}

template <typename T>
inline double calculate_mse(const dal::table& infer_labels, const T* ground_truth) {
    double mean = 0.0;
    const auto labels = dal::row_accessor<const T>(infer_labels).pull();
    for (std::int64_t i = 0; i < labels.get_count(); i++) {
        double val = (labels[i] - ground_truth[i]) * (labels[i] - ground_truth[i]);
        mean += (val - mean) / static_cast<double>(i + 1);
    }

    return mean;
}

template <typename T>
inline double calculate_mse(const dal::table& infer_labels, const dal::table& ground_truth) {
    double mean = 0.0;
    const auto labels = dal::row_accessor<const T>(infer_labels).pull();
    const auto truth_labels = dal::row_accessor<const T>(ground_truth).pull();
    for (std::int64_t i = 0; i < labels.get_count(); i++) {
        double val = (labels[i] - truth_labels[i]) * (labels[i] - truth_labels[i]);
        mean += (val - mean) / static_cast<double>(i + 1);
    }

    return mean;
}

inline void verify_oob_err_vs_oob_err_per_observation(const dal::table& oob_err,
                                                      const dal::table& oob_err_per_observation,
                                                      double threshold) {
    const auto oob_err_val = dal::row_accessor<const double>(oob_err).pull();
    const auto oob_err_per_obs_arr =
        dal::row_accessor<const double>(oob_err_per_observation).pull();

    std::int64_t oob_err_obs_count = 0;
    double ref_oob_err = 0.0;
    for (std::int64_t i = 0; i < oob_err_per_obs_arr.get_count(); i++) {
        if (oob_err_per_obs_arr[i] >= 0.0) {
            oob_err_obs_count++;
            ref_oob_err += oob_err_per_obs_arr[i];
        }
        else {
            ASSERT_GE(oob_err_per_obs_arr[i], -1.0);
        }
    }

    ASSERT_LE(((oob_err_val[0] - ref_oob_err) / oob_err_val[0]), threshold);
}
