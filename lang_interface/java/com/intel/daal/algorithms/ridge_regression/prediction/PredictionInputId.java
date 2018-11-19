/* file: PredictionInputId.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup ridge_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.ridge_regression.prediction;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__PREDICTION__PREDICTIONINPUTID"></a>
 * @brief Available identifiers of input objects for ridge regression model-based prediction
 */
public final class PredictionInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public PredictionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int dataId  = 0;
    private static final int modelId = 1;

    /** %Input data table */
    public static final PredictionInputId data  = new PredictionInputId(dataId);
    /** Trained ridge regression model */
    public static final PredictionInputId model = new PredictionInputId(modelId);
}
/** @} */
