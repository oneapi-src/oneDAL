/* file: PartialSVDCollectionResultID.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALSVDCOLLECTIONRESULTID"></a>
 * @brief Available identifiers of partial results of the %SVD method of the PCA algorithm
 */
public final class PartialSVDCollectionResultID {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public PartialSVDCollectionResultID(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int svdAuxiliaryDataId = 3;

    /*!< Auxiliary data */
    public static final PartialSVDCollectionResultID svdAuxiliaryData = new PartialSVDCollectionResultID(
            svdAuxiliaryDataId);
}
/** @} */
