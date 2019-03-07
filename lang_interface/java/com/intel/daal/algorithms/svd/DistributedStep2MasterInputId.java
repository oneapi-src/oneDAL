/* file: DistributedStep2MasterInputId.java */
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
 * @ingroup svd_distributed
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP2MASTERINPUTID"></a>
 * @brief Partial results from previous steps of the SVD algorithm in the distributed processing mode, required by the second step
 */
public final class DistributedStep2MasterInputId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the master node input object identifier using the provided value
     * @param value     Value corresponding to the master node input object identifier
     */
    public DistributedStep2MasterInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the master node input object identifier
     * @return Value corresponding to the master node input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int inputOfStep2FromStep1Id = 0;

    public static final DistributedStep2MasterInputId inputOfStep2FromStep1 = new DistributedStep2MasterInputId(
            inputOfStep2FromStep1Id); /*!< DataCollection with data transferred from the first step to the second step in the distributed
                                         * processing mode */
}
/** @} */
