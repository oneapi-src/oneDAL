/* file: TransformParameter.java */
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
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__PARAMETER"></a>
 * @brief Parameters of the PCA transformation algorithm
 */
public class TransformParameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter of PCA transformation algorithm
     *
     * @param context                  Context to manage the PCA transformation algorithm
     * @param cObject                  Address of C++ parameter
     */
    public TransformParameter(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Gets the number of components for PCA transformation.
     * @return  The the number of components for PCA transformation.
     */
    public long getNumberOfComponents() {
        return cGetNumberOfComponents(this.cObject);
    }

    /**
     * Sets the the number of components for PCA transformation.
     * @param nComponents The number of components for PCA transformation.
     */
    public void setNumberOfComponents(long nComponents) {
        cSetNumberOfComponents(this.cObject, nComponents);
    }

    private native void cSetNumberOfComponents(long cObject, long nComponents);
    private native long cGetNumberOfComponents(long cObject);
}
/** @} */
