/* file: InitDistributedStep3MasterPlusPlusInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP3MASTERPLUSPLUSINPUTID"></a>
 * @brief Available identifiers of input objects for computing initial clusters for the K-Means algorithm
 *        used with plusPlus and parallelPlus methods only on the 3rd step on a master node.
 */
public final class InitDistributedStep3MasterPlusPlusInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InitDistributedStep3MasterPlusPlusInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int inputOfStep3FromStep2Value = 0;

    /** Numeric table with the data calculated on step2 on local nodes */
    public static final InitDistributedStep3MasterPlusPlusInputId inputOfStep3FromStep2 = new InitDistributedStep3MasterPlusPlusInputId(
            inputOfStep3FromStep2Value);
}
/** @} */
