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
 * @ingroup prediction
 * @{
 */
package com.intel.daal.algorithms.classifier.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__PREDICTIONRESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of the classifier model-based prediction algorithm in the batch processing mode
 */

public class PredictionResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the classifier model-based prediction algorithm
     * @param context   Context to manage the result of the classifier model-based prediction algorithm
     */
    public PredictionResult(DaalContext context) {
        super(context);
        cObject = cNewResult();
    }

    public PredictionResult(DaalContext context, long cAlgorithm) {
        super(context);
        cObject = cGetResult(cAlgorithm);
    }

    /**
     * Returns the final result of the classification algorithm
     * @param id   Identifier of the result, @ref PredictionResultId
     * @return     Result that corresponds to the given identifier
     */
    public NumericTable get(PredictionResultId id) {
        PredictionResultId.throwIfInvalid(id);
        return (NumericTable)Factory.instance().createObject(getContext(),
                cGetResultTable(cObject, id.getValue()));
    }

    /**
     * Sets the final result of the algorithm
     * @param id    Identifier of the final result
     * @param value Object for storing the final result
     */
    public void set(PredictionResultId id, NumericTable value) {
        PredictionResultId.throwIfInvalid(id);
        cSetResultTable(cObject, id.getValue(), value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResult(long algAddress);

    private native long cGetResultTable(long resAddr, int id);

    private native void cSetResultTable(long cResult, int id, long cNumericTable);
}
/** @} */
