/* file: DistributedPartialResultCollectionId.java */
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
 * @defgroup qr_distributed Distributed
 * @ingroup qr_without_pivoting
 * @{
 */
package com.intel.daal.algorithms.qr;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDPARTIALRESULTCOLLECTIONID"></a>
 * @brief Available types of partial results of the QR decomposition algorithm on the second step in the distributed processing mode */
public final class DistributedPartialResultCollectionId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultCollectionId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int outputOfStep2ForStep3Id = 0;

    public static final DistributedPartialResultCollectionId outputOfStep2ForStep3 = new DistributedPartialResultCollectionId(
            outputOfStep2ForStep3Id); /*!< DataCollection with data to be transfered to distributed step 3 */
}
/** @} */
