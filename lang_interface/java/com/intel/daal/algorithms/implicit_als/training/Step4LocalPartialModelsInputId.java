/* file: Step4LocalPartialModelsInputId.java */
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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP4LOCALPARTIALMODELSINPUTID"></a>
 * @brief Available identifiers of input objects for the implicit ALS training algorithm
 * in the fourth step of the distributed processing mode
 */
public final class Step4LocalPartialModelsInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the local input object identifier using the provided value
     * @param value     Value corresponding to the local input object identifier
     */
    public Step4LocalPartialModelsInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the local input object identifier
     * @return Value corresponding to the local input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partialModelsId = 0;

    /** %Input partial models for the implicit ALS training algorithm in the fourth
    * step of the distributed processing mode
    */
    public static final Step4LocalPartialModelsInputId partialModels =
            new Step4LocalPartialModelsInputId(partialModelsId);
}
/** @} */
