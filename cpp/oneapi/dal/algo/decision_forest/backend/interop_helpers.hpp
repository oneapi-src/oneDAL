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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/detail/model_impl.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

#include <daal/include/algorithms/decision_forest/decision_forest_classification_model.h>
#include <daal/include/algorithms/decision_forest/decision_forest_regression_model.h>

#include <daal/include/algorithms/decision_forest/decision_forest_training_parameter.h>
#include <daal/src/algorithms/dtrees/forest/classification/df_classification_predict_dense_default_batch.h>

namespace oneapi::dal::backend::interop::decision_forest {

namespace daal_df = daal::algorithms::decision_forest;
namespace cls = daal::algorithms::decision_forest::classification;

namespace df = dal::decision_forest;
namespace df_interop = dal::backend::interop::decision_forest;

inline auto convert_to_daal_voting_mode(df::voting_mode vm) {
    return df::voting_mode::weighted == vm ? cls::prediction::weighted
                                           : cls::prediction::unweighted;
}

inline auto convert_to_daal_variable_importance_mode(df::variable_importance_mode vimp) {
    return df::variable_importance_mode::mdi == vimp
               ? daal_df::training::MDI
               : df::variable_importance_mode::mda_raw == vimp
                     ? daal_df::training::MDA_Raw
                     : df::variable_importance_mode::mda_scaled == vimp
                           ? daal_df::training::MDA_Scaled
                           : daal_df::training::none;
}

/* oneDal -> daal model bridge */
template <typename T>
inline typename std::enable_if_t<std::is_same_v<T, std::decay_t<daal_df::classification::ModelPtr>>,
                                 std::int64_t>
get_number_of_classes(T model) {
    return model->getNumberOfClasses();
}

template <typename T>
inline
    typename std::enable_if_t<!std::is_same_v<T, std::decay_t<daal_df::classification::ModelPtr>>,
                              std::int64_t>
    get_number_of_classes(T model) {
    return 0;
}

template <typename Task, typename ModelP = daal_df::classification::ModelPtr>
class interop_model_impl : public df::detail::model_impl<Task> {
public:
    interop_model_impl() = delete;
    interop_model_impl(ModelP pmdl) : pmdl_(pmdl){};
    virtual ~interop_model_impl(){};

    virtual std::int64_t get_tree_count() const {
        return pmdl_->numberOfTrees();
    }
    virtual std::int64_t get_class_count() const {
        return get_number_of_classes<ModelP>(pmdl_);
    }
    void clear() {
        pmdl_->clear();
    }

    virtual bool is_interop() const {
        return true;
    }

    void set_model(ModelP pmdl) {
        pmdl_ = pmdl;
    }
    ModelP get_model() {
        return pmdl_;
    }

private:
    ModelP pmdl_;
};

} // namespace oneapi::dal::backend::interop::decision_forest
