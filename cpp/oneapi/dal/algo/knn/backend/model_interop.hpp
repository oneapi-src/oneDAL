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

#include <algorithms/classifier/classifier_model.h>
#include <daal/include/algorithms/k_nearest_neighbors/bf_knn_classification_model.h>
#include <daal/include/algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h>
#include "oneapi/dal/algo/knn/common.hpp"

namespace oneapi::dal::knn::backend {

inline auto convert_to_daal_bf_voting_mode(voting_mode vm) {
    namespace daal_bf_knn = daal::algorithms::bf_knn_classification;
    return voting_mode::uniform == vm ? daal_bf_knn::voteUniform : daal_bf_knn::voteDistance;
}

inline auto convert_to_daal_kdtree_voting_mode(voting_mode vm) {
    namespace daal_kdtree_knn = daal::algorithms::kdtree_knn_classification;
    return voting_mode::uniform == vm ? daal_kdtree_knn::voteUniform
                                      : daal_kdtree_knn::voteDistance;
}

class model_interop : public base {
public:
    using DaalModel = daal::algorithms::classifier::ModelPtr;

    model_interop(const DaalModel& daal_model) : daal_model_(daal_model) {}

    void set_daal_model(const DaalModel& model) {
        daal_model_ = model;
    }

    const DaalModel& get_daal_model() const {
        return daal_model_;
    }

private:
    DaalModel daal_model_;
};

} // namespace oneapi::dal::knn::backend
