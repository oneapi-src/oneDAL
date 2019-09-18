/* file: PredictionInput.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/**
 * @defgroup brownboost_prediction Prediction
 * @brief Contains classes for prediction based on BrownBoost models
 * @ingroup brownboost
 * @{
 */
package com.intel.daal.algorithms.brownboost.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.brownboost.Model;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__PREDICTION__PREDICTIONINPUT"></a>
 * @brief  %Input objects for the BrownBoost algorithm
 */
public class PredictionInput extends com.intel.daal.algorithms.classifier.prediction.PredictionInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PredictionInput(DaalContext context, long cAlgorithm) {
        super(context, cAlgorithm);
    }

    public PredictionInput(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context, cAlgorithm, cmode);
    }

    /**
     * Returns the Model input object for the BrownBoost model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Model get(ModelInputId id) {
        ModelInputId.throwIfInvalid(id);
        return new Model(getContext(), cGetInputModel(cObject, id.getValue()));
    }
}
/** @} */
