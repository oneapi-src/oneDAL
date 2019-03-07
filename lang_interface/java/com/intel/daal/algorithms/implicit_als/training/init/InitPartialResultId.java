/* file: InitPartialResultId.java */
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
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITPARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of the default initialization
 * of the implicit ALS training algorithm
 */
public final class InitPartialResultId {
    private int _value;

    /**
     * Constructs the initialization partial result object identifier using the provided value
     * @param value     Value corresponding to the initialization partial result object identifier
     */
    public InitPartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization partial result object identifier
     * @return Value corresponding to the initialization partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partialModelId = 2;

    /** Partial model trained on the available input data */
    public static final InitPartialResultId partialModel = new InitPartialResultId(
            partialModelId);
}
/** @} */
