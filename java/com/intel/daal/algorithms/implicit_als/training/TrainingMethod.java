/* file: TrainingMethod.java */
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
 * @ingroup implicit_als_training
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for training the implicit ALS model
 */
public final class TrainingMethod {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the training method object using the provided value
     * @param value     Value corresponding to the training method object
     */
    public TrainingMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training method object
     * @return Value corresponding to the training method object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int defaultDenseId = 0;
    @Native private static final int fastCSRId      = 1;

    /**
    * Method for training the implicit ALS model
    */
    public static final TrainingMethod defaultDense = new TrainingMethod(
            defaultDenseId);    /*!< Default: method proposed by Hu, Koren,
                                    Volinsky for input data stored in the dense format */
    /**
    * Method for training the implicit ALS model
    */
    public static final TrainingMethod fastCSR      = new TrainingMethod(
            fastCSRId);         /*!< Method proposed by Hu, Koren,
                                    Volinsky for input data stored in the compressed sparse row (CSR) format */
}
/** @} */
