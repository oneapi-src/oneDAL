/* file: InitPartialResultCollectionId.java */
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
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITPARTIALRESULTCOLLECTIONID"></a>
 * @brief Available identifiers of partial results of the implicit ALS initialization algorithm
 *        in the first step of the distributed processing mode
 */
public final class InitPartialResultCollectionId {
    private int _value;

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public InitPartialResultCollectionId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int OutputOfStep1ForStep2 = 3;

    /**
     * Partial results of the implicit ALS initialization algorithm computed in the first step
     * and to be transferred to the second step of the distributed initialization algorithm
     */
    public static final InitPartialResultCollectionId outputOfStep1ForStep2 = new InitPartialResultCollectionId(
            OutputOfStep1ForStep2);
}
/** @} */
