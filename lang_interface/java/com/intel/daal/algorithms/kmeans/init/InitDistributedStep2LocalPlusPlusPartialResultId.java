/* file: InitDistributedStep2LocalPlusPlusPartialResultId.java */
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
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2LOCALPLUSPLUSPARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of computing initial clusters for the K-Means algorithm in the distributed processing mode
 *        used with plusPlus and parallelPlus methods only on the 2nd step on a local node.
 */
public final class InitDistributedStep2LocalPlusPlusPartialResultId {
    private int _value;

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public InitDistributedStep2LocalPlusPlusPartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int outputOfStep2ForStep3Value = 0;
    private static final int outputOfStep2ForStep5Value = 1;

    /** Numeric table containing output from step 2 on the local node used by step 3 on a master node */
    public static final InitDistributedStep2LocalPlusPlusPartialResultId outputOfStep2ForStep3 = new InitDistributedStep2LocalPlusPlusPartialResultId(
            outputOfStep2ForStep3Value);
    /** Numeric table containing output from step 2 on the local node used by step 3 on a master node */
    public static final InitDistributedStep2LocalPlusPlusPartialResultId outputOfStep2ForStep5 = new InitDistributedStep2LocalPlusPlusPartialResultId(
            outputOfStep2ForStep5Value);
}
/** @} */
