/* file: MasterInputId.java */
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
 * @ingroup pca
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__MASTERINPUTID"></a>
 * @brief Available identifiers of the PCA algorithm on the second step in the distributed processing mode
 */
public final class MasterInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the master input object identifier using the provided value
     * @param value     Value corresponding to the master input object identifier
     */
    public MasterInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the master input object identifier
     * @return Value corresponding to the master input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partialResultsId = 0;

    /** Partial results computed by the PCA algorithm on the first step in the distributed processing mode */
    public static final MasterInputId partialResults = new MasterInputId(partialResultsId);
}
/** @} */
