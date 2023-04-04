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

#include <daal/include/algorithms/decision_forest/decision_forest_classification_predict_types.h>
#include <daal/include/algorithms/decision_forest/decision_forest_classification_training_types.h>
#include <daal/include/algorithms/decision_forest/decision_forest_regression_predict_types.h>
#include <daal/include/algorithms/decision_forest/decision_forest_regression_training_types.h>

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/backend/serialization.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/archive.hpp"

namespace oneapi::dal::decision_forest::backend {

inline auto convert_to_daal_voting_mode(voting_mode vm) {
    namespace daal_df_cls = daal::algorithms::decision_forest::classification;
    return voting_mode::weighted == vm ? daal_df_cls::prediction::weighted
                                       : daal_df_cls::prediction::unweighted;
}

inline auto convert_to_daal_variable_importance_mode(variable_importance_mode vimp) {
    namespace daal_df = daal::algorithms::decision_forest;
    return variable_importance_mode::mdi == vimp          ? daal_df::training::MDI
           : variable_importance_mode::mda_raw == vimp    ? daal_df::training::MDA_Raw
           : variable_importance_mode::mda_scaled == vimp ? daal_df::training::MDA_Scaled
                                                          : daal_df::training::none;
}

inline auto convert_to_daal_splitter_mode(splitter_mode splitter) {
    namespace daal_df = daal::algorithms::decision_forest;
    return splitter_mode::best == splitter ? daal_df::training::best : daal_df::training::random;
}

class model_interop : public base {
public:
    virtual ~model_interop() = default;
    virtual void clear() {}
};

#define DF_MODEL_INTEROP_SERIALIZABLE(model_t, ClassificationId, RegressionId)           \
    ONEDAL_SERIALIZABLE_MAP2(                                                            \
        model_t,                                                                         \
        (daal::algorithms::decision_forest::classification::ModelPtr, ClassificationId), \
        (daal::algorithms::decision_forest::regression::ModelPtr, RegressionId))

template <typename model_ptr_t>
class model_interop_impl
        : public model_interop,
          public DF_MODEL_INTEROP_SERIALIZABLE(model_ptr_t,
                                               decision_forest_model_interop_impl_cls_id,
                                               decision_forest_model_interop_impl_reg_id) {
public:
    model_interop_impl() = default;

    model_interop_impl(const model_ptr_t& model) : daal_model_(model) {}

    const model_ptr_t get_model() const {
        return daal_model_;
    }

    void clear() override {
        daal_model_->clear();
    }

    void serialize(dal::detail::output_archive& ar) const override {
        dal::backend::interop::daal_output_data_archive daal_ar(ar);
        daal_ar.setSharedPtrObj(const_cast<model_ptr_t&>(daal_model_));
    }

    void deserialize(dal::detail::input_archive& ar) override {
        dal::backend::interop::daal_input_data_archive daal_ar(ar);
        daal_ar.setSharedPtrObj(daal_model_);
    }

private:
    model_ptr_t daal_model_;
};

using model_interop_cls =
    model_interop_impl<daal::algorithms::decision_forest::classification::ModelPtr>;

using model_interop_reg =
    model_interop_impl<daal::algorithms::decision_forest::regression::ModelPtr>;

} // namespace oneapi::dal::decision_forest::backend
