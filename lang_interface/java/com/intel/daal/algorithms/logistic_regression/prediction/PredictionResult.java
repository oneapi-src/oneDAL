/* file: PredictionResult.java */
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
