/* file: PredictionInput.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @defgroup stump_prediction Prediction
 * @brief Contains classes to make prediction based on the decision stump model
 * @ingroup stump
 * @{
 */
package com.intel.daal.algorithms.stump.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.weak_learner.Model;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__PREDICTION__PREDICTIONINPUT"></a>
 * @brief  %Input objects for the Stump algorithm
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
     * Returns the Model input object for the Stump model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Model get(ModelInputId id) {
        if (id == ModelInputId.model) {
            return new Model(getContext(), cGetInputModel(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }
}
/** @} */
