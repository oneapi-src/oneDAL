/* file: DistributedPartialResultStep4Id.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4ID"></a>
 * @brief Available identifiers of partial results of the implicit ALS training algorithm obtained
 * in the fourth step of the distributed processing mode
 */
public final class DistributedPartialResultStep4Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep4Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int outputOfStep4ForStep1Id = 0;
    private static final int outputOfStep4ForStep3Id = 0;
    private static final int outputOfStep4Id = 0;

/** Partial results of the implicit ALS training algorithm obtained in the fourth step
*   and to be transferred to the first step of the distributed processing mode
*/
    public static final DistributedPartialResultStep4Id outputOfStep4ForStep1 = new DistributedPartialResultStep4Id(
            outputOfStep4ForStep1Id);

/** Partial results of the implicit ALS training algorithm obtained in the fourth step
*   and to be transferred to the third step of the distributed processing mode
*/
    public static final DistributedPartialResultStep4Id outputOfStep4ForStep3 = new DistributedPartialResultStep4Id(
            outputOfStep4ForStep3Id);
/** Partial results of the implicit ALS training algorithm obtained in the fourth step of the distributed processing mode */
    public static final DistributedPartialResultStep4Id outputOfStep4 = new DistributedPartialResultStep4Id(
            outputOfStep4Id);
}
/** @} */
