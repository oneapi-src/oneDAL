/* file: InitPartialResultBaseId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITPARTIALRESULTBASEID"></a>
 * @brief Available identifiers of partial results of the implicit ALS initialization algorithm
 *        in the first and second steps of the distributed processing mode
 */
public final class InitPartialResultBaseId {
    private int _value;

    /**
     * Constructs the initialization partial result object identifier using the provided value
     * @param value     Value corresponding to the initialization partial result object identifier
     */
    public InitPartialResultBaseId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization partial result object identifier
     * @return Value corresponding to the initialization partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int OutputOfInitForComputeStep3 = 0;
    private static final int Offsets = 1;

    /**
     * Partial results of the implicit ALS initialization algorithm computed in the first step
     * and to be transferred to the third step of the distributed ALS training algorithm
     */
    public static final InitPartialResultBaseId outputOfInitForComputeStep3 = new InitPartialResultBaseId(OutputOfInitForComputeStep3);
    /**
     * Collection of 1x1 numeric tables that hold the global indices of the starting rows
     * of the partial models
     */
    public static final InitPartialResultBaseId offsets = new InitPartialResultBaseId(Offsets);
}
/** @} */
