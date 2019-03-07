/* file: TransformInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__INPUTID"></a>
 * \brief Available identifiers of input objects for the PCA transformation algorithm
 */
public final class TransformInputId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public TransformInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int InputDataId = 0;
    private static final int InputEigenvectorsId = 1;

    public static final TransformInputId data = new TransformInputId(InputDataId); /*!< %Input data table */
    public static final TransformInputId eigenvectors = new TransformInputId(InputEigenvectorsId); /*!< %Input eigenvectors table */
}
/** @} */
