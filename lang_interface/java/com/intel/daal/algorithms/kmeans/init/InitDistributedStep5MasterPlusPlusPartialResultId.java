/* file: InitDistributedStep5MasterPlusPlusPartialResultId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP5MASTERPLUSPLUSPARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of computing initial clusters for the K-Means algorithm in the distributed processing mode
 *        used with parallelPlus method only on the 5th step on a master node.
 */
public final class InitDistributedStep5MasterPlusPlusPartialResultId {
    private int _value;

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public InitDistributedStep5MasterPlusPlusPartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int candidatesValue = 0;
    private static final int weightsValue = 0;

    /** NumericTable with the new centroids calculated on the previous steps */
    public static final InitDistributedStep5MasterPlusPlusPartialResultId candidates =
        new InitDistributedStep5MasterPlusPlusPartialResultId(candidatesValue);
    /** NumericTable with the weights of the new centroids calculated on the previous steps */
    public static final InitDistributedStep5MasterPlusPlusPartialResultId weights =
        new InitDistributedStep5MasterPlusPlusPartialResultId(weightsValue);
}
/** @} */
