/* file: TransformComponentId.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

import java.lang.annotation.Native;

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

    @Native private static final int TransformComponentMeansId = 1;
    @Native private static final int TransformComponentVariancesId = 2;
    @Native private static final int TransformComponentEigenvaluesId = 4;

    public static final TransformComponentId mean = new TransformComponentId(TransformComponentMeansId); /*!< %means id */
    public static final TransformComponentId variance = new TransformComponentId(TransformComponentVariancesId); /*!< %variances id */
    public static final TransformComponentId eigenvalue = new TransformComponentId(TransformComponentEigenvaluesId); /*!< %eigenvalues id */
}
/** @} */
