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
 * @ingroup linear_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.linear_regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__PREDICTIONRESULT"></a>
 * @brief Result object for linear regression model-based prediction
*/
public final class PredictionResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the linear regression prediction result
     * @param context   Context to manage the algorithm
     */
    public PredictionResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public PredictionResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of linear regression model-based prediction
     * @param id    Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public NumericTable get(PredictionResultId id) {
        int idValue = id.getValue();
        if (idValue != PredictionResultId.prediction.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }

        return (NumericTable)Factory.instance().createObject(getContext(), cGetPredictionResult(cObject, idValue));
    }

    /**
     * Sets the result of linear regression model-based prediction
     * @param id    Identifier of the result
     * @param val   Result that corresponds to the given identifier
     */
    public void set(PredictionResultId id, NumericTable val) {
        int idValue = id.getValue();
        if (idValue != PredictionResultId.prediction.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPredictionResult(cObject, idValue, val.getCObject());
    }

    private native long cNewResult();

    private native long cGetPredictionResult(long resAddr, int id);

    private native void cSetPredictionResult(long cObject, int id, long cNumericTable);

}
/** @} */
