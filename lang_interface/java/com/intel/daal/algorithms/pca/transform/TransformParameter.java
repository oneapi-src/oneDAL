/* file: TransformParameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
