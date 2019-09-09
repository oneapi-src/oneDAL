/* file: PredictionResult.java */
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
 * @ingroup logistic_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.logistic_regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__PREDICTIONRESULT"></a>
 * @brief Result object for logistic regression model-based prediction
 */
public final class PredictionResult extends com.intel.daal.algorithms.classifier.prediction.PredictionResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the logistic regression prediction result
     * @param context   Context to manage the  result of the logistic regression prediction algorithm
     */
    public PredictionResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public PredictionResult(DaalContext context, long cResult) {
        super(context);
        this.cObject = cResult;
    }

    /**
     * Returns the result of logistic regression model-based prediction
     * @param id    Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public NumericTable get(PredictionResultNumericTableId id) {
        if (id != PredictionResultNumericTableId.probabilities && id != PredictionResultNumericTableId.logProbabilities) {
            throw new IllegalArgumentException("id unsupported");
        }

        return (NumericTable)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
    }

    /**
     * Sets the result of logistic regression model-based prediction
     * @param id    Identifier of the result
     * @param val   Result that corresponds to the given identifier
     */
    public void set(PredictionResultNumericTableId id, NumericTable val) {
        if (id != PredictionResultNumericTableId.probabilities && id != PredictionResultNumericTableId.logProbabilities) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetValue(cObject, id.getValue(), val.getCObject());
    }

    private native long cNewResult();
    private native long cInit(long algaddr);
    private native long cGetValue(long resAddr, int id);
    private native void cSetValue(long cObject, int id, long cNumericTable);

}
/** @} */
