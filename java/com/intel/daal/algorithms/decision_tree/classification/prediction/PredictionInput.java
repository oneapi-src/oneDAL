/* file: PredictionInput.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @defgroup decision_tree_classification_prediction Prediction
 * @brief Contains classes for prediction based on decision tree classification models
 * @ingroup decision_tree_classification
 * @{
 */
package com.intel.daal.algorithms.decision_tree.classification.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_tree.classification.Model;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PREDICTION__PREDICTIONINPUT"></a>
 * @brief  %Input objects for the decision tree classification algorithm
 */
public class PredictionInput extends com.intel.daal.algorithms.classifier.prediction.PredictionInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PredictionInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Returns the Model input object for the decision tree classification model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    @Override
    public Model get(ModelInputId id) {
        if (id == ModelInputId.model) {
            return new Model(getContext(), cGetInputModel(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }
}
/** @} */
