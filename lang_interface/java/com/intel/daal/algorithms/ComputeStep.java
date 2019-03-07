/* file: ComputeStep.java */
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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COMPUTESTEP"></a>
 * Describes on which node the computation stage is done in distributed computations
 */
public final class ComputeStep {
    private int _value;

    /**
     * Constructs the compute step object using the provided value
     * @param value     Value corresponding to the compute step object
     */
    public ComputeStep(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the compute step object
     * @return Value corresponding to the compute step object
     */
    public int getValue() {
        return _value;
    }

    private static final int step1LocalValue  = 0;
    private static final int step2MasterValue = 1;
    private static final int step3LocalValue  = 2;

    /** Processing is done on local nodes */
    public static final ComputeStep step1Local  = new ComputeStep(step1LocalValue);
    /** Processing is done on master nodes */
    public static final ComputeStep step2Master = new ComputeStep(step2MasterValue);
    /** Finalization is done on local nodes */
    public static final ComputeStep step3Local  = new ComputeStep(step3LocalValue);
}
/** @} */
