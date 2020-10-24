/* file: PredictionResult.java */
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

/**
 * @ingroup multi-class_classifier
 * @{
 */
package com.intel.daal.algorithms.multi_class_classifier.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__PREDICTIONRESULT"></a>
 * @brief Result object for multi-class classifier model-based prediction
 */
public final class PredictionResult extends com.intel.daal.algorithms.classifier.prediction.PredictionResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the multi-class classifier prediction result
     * @param context   Context to manage the  result of the multi-class classifier prediction algorithm
     */
    public PredictionResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public PredictionResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of multi-class classifier model-based prediction
     * @param id    Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public NumericTable get(PredictionResultId id) {
        int idValue = id.getValue();
        if (idValue != PredictionResultId.prediction.getValue() && idValue != PredictionResultId.decisionFunction.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }

        return (NumericTable)Factory.instance().createObject(getContext(), cGetPredictionResult(cObject, idValue));
    }

    /**
     * Sets the result of multi-class classifier model-based prediction
     * @param id    Identifier of the result
     * @param val   Result that corresponds to the given identifier
     */
    public void set(PredictionResultId id, NumericTable val) {
        int idValue = id.getValue();
        if (idValue != PredictionResultId.prediction.getValue() && idValue != PredictionResultId.decisionFunction.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPredictionResult(cObject, idValue, val.getCObject());
    }

    private native long cNewResult();

    private native long cGetPredictionResult(long resAddr, int id);

    private native void cSetPredictionResult(long cObject, int id, long cNumericTable);

}
/** @} */
