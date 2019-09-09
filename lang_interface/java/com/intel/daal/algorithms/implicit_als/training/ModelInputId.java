/* file: ModelInputId.java */
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
 * @defgroup implicit_als_training Training
 * @brief Contains classes of the implicit ALS training algorithm
 * @ingroup implicit_als
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__MODELINPUTID"></a>
 * @brief Available identifiers of input model objects for the implicit ALS training
 * algorithm
 */
public final class ModelInputId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the model input object identifier using the provided value
     * @param value     Value corresponding to the model input object identifier
     */
    public ModelInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the model input object identifier
     * @return Value corresponding to the model input object identifier
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int inputModelId = 1;

    /** Input model */
    public static final ModelInputId inputModel = new ModelInputId(
            inputModelId); /*!< Initial model that contains
                                              initialized factors */
}
/** @} */
