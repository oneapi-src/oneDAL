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
 * @ingroup decision_tree_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.decision_tree.regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_tree.regression.prediction.PredictionResultId;
import com.intel.daal.algorithms.Precision;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__PREDICTIONRESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of
 *        the decision_tree regression model-based prediction algorithm in the batch processing mode
 */
public final class PredictionResult extends com.intel.daal.algorithms.regression.prediction.PredictionResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** Default constructor */
    public PredictionResult(DaalContext context, long cAlgorithm) {
        super(context, cAlgorithm);
    }

    /**
     * Returns the final result of the regression algorithm
     * @param id   Identifier of the result, @ref PredictionResultId
     * @return     Result that corresponds to the given identifier
     */
    public NumericTable get(PredictionResultId id) {
        if (id == PredictionResultId.prediction) {
            return (NumericTable)Factory.instance().createObject(getContext(),
                    cGetResultTable(cObject, PredictionResultId.prediction.getValue()));
        } else {
            return null;
        }
    }

    /**
     * Sets the final result of the algorithm
     * @param id    Identifier of the final result
     * @param value Object for storing the final result
     */
    public void set(PredictionResultId id, NumericTable value) {
        if (id != PredictionResultId.prediction) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, id.getValue(), value.getCObject());
    }

    private native long cGetResultTable(long resAddr, int id);

    private native void cSetResultTable(long resAddr, int id, long ntAddr);
}
/** @} */
