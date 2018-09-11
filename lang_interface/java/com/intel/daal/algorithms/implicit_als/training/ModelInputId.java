/* file: ModelInputId.java */
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
 * @defgroup implicit_als_training Training
 * @brief Contains classes of the implicit ALS training algorithm
 * @ingroup implicit_als
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__MODELINPUTID"></a>
 * @brief Available identifiers of input model objects for the implicit ALS training
 * algorithm
 */
public final class ModelInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the model input object identifier using the provided value
     * @param value     Value corresponding to the model input object identifier
     */
    public ModelInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the model input object identifier
     * @return Value corresponding to the model input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int inputModelId = 1;

    /** Input model */
    public static final ModelInputId inputModel = new ModelInputId(
            inputModelId); /*!< Initial model that contains
                                              initialized factors */
}
/** @} */
