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

#pragma once

#include <algorithms/classifier/classifier_model.h>
#include <daal/include/algorithms/k_nearest_neighbors/bf_knn_classification_model.h>
#include <daal/include/algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h>
#include "oneapi/dal/algo/knn/common.hpp"

#include "oneapi/dal/backend/serialization.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/archive.hpp"

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

class model_interop : public ONEDAL_SERIALIZABLE(knn_model_interop_id) {
    using daal_model_ptr_t = daal::algorithms::classifier::ModelPtr;

public:
    model_interop() = default;

    model_interop(const daal_model_ptr_t& daal_model) : daal_model_(daal_model) {}

    void set_daal_model(const daal_model_ptr_t& model) {
        daal_model_ = model;
    }

    const daal_model_ptr_t& get_daal_model() const {
        return daal_model_;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        dal::backend::interop::daal_output_data_archive daal_ar(ar);
        daal_ar.setSharedPtrObj(const_cast<daal_model_ptr_t&>(daal_model_));
    }

    void deserialize(dal::detail::input_archive& ar) override {
        dal::backend::interop::daal_input_data_archive daal_ar(ar);
        daal_ar.setSharedPtrObj(daal_model_);
    }

private:
    daal_model_ptr_t daal_model_;
};

} // namespace oneapi::dal::knn::backend
