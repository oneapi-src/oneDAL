/* file: InitDistributedStep5MasterPlusPlusInputId.java */
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
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP5MASTERPLUSPLUSINPUTID"></a>
 * @brief Available identifiers of input objects for computing initial clusters for the K-Means algorithm
 *        used with plusPlus and parallelPlus methods only on the 5th step on a master node.
 */
public final class InitDistributedStep5MasterPlusPlusInputId {
    private int _value;

    /**
     * Constructs the initialization input object identifier using the provided value
     * @param value     Value corresponding to the initialization input object identifier
     */
    public InitDistributedStep5MasterPlusPlusInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization input object identifier
     * @return Value corresponding to the initialization input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int inputCentroidsValue = 0;
    private static final int inputOfStep5FromStep2Value = 1;

    /** DataCollection of NumericTables with the new centroids */
    public static final InitDistributedStep5MasterPlusPlusInputId inputCentroids = new InitDistributedStep5MasterPlusPlusInputId(
            inputCentroidsValue);
    /** DataCollection of NumericTables with the new centroids rating */
    public static final InitDistributedStep5MasterPlusPlusInputId inputOfStep5FromStep2 = new InitDistributedStep5MasterPlusPlusInputId(
            inputOfStep5FromStep2Value);
}
/** @} */
