/* file: DistributedStep3LocalInputId.java */
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
 * @ingroup qr_distributed
 * @{
 */
package com.intel.daal.algorithms.qr;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP3LOCALINPUTID"></a>
 * @brief Partial results required by the QR decomposition algorithm on the third step in the distributed processing mode
 */
public final class DistributedStep3LocalInputId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the local input object identifier using the provided value
     * @param value     Value corresponding to the local input object identifier
     */
    public DistributedStep3LocalInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the local input object identifier
     * @return Value corresponding to the local input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int inputOfStep3FromStep1Id = 0;
    private static final int inputOfStep3FromStep2Id = 1;

    /** DataCollection with data transferred from the first step to the third step in the distributed processing mode */
    public static final DistributedStep3LocalInputId inputOfStep3FromStep1 = new DistributedStep3LocalInputId(
            inputOfStep3FromStep1Id);
    /** DataCollection with data transferred from the second step to the third step in the distributed processing mode*/
    public static final DistributedStep3LocalInputId inputOfStep3FromStep2 = new DistributedStep3LocalInputId(
            inputOfStep3FromStep2Id);
}
/** @} */
