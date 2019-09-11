/* file: MasterInputId.java */
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
 * @ingroup linear_regression_training
 * @{
 */
package com.intel.daal.algorithms.linear_regression.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__MASTERINPUTID"></a>
 * @brief Available identifiers of input objects for linear regression model-based training on the master node
 */
public final class MasterInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public MasterInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partialModelsId = 0;

    /** Partial models trained on local nodes */
    public static final MasterInputId partialModels = new MasterInputId(partialModelsId);
}
/** @} */
