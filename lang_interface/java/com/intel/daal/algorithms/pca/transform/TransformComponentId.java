/* file: TransformComponentId.java */
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
 * @ingroup pca_transform
 * @{
 */
package com.intel.daal.algorithms.pca.transform;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__TRANSFORMCOMPONENTID"></a>
 * \brief Available identifiers of input objects for the PCA transformation algorithm
 */
public final class TransformComponentId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public TransformComponentId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int TransformComponentMeansId = 1;
    private static final int TransformComponentVariancesId = 2;
    private static final int TransformComponentEigenvaluesId = 4;

    public static final TransformComponentId mean = new TransformComponentId(TransformComponentMeansId); /*!< %means id */
    public static final TransformComponentId variance = new TransformComponentId(TransformComponentVariancesId); /*!< %variances id */
    public static final TransformComponentId eigenvalue = new TransformComponentId(TransformComponentEigenvaluesId); /*!< %eigenvalues id */
}
/** @} */
