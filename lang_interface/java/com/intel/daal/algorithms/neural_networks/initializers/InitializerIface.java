/* file: InitializerIface.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
 * @brief Contains classes for the neural network weights and biases initializers
 */
package com.intel.daal.algorithms.neural_networks.initializers;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__INITIALIZERIFACE"></a>
 * @brief Class representing a neural network weights and biases initializer
 *
 * @par References
 *      - Input class
 */
public abstract class InitializerIface extends com.intel.daal.algorithms.AnalysisBatch {
    public Input input;     /*!< %Input of the initializer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs initializer algorithm
     * @param context Context to manage the initializer
     */
    public InitializerIface(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated initializer with a copy of input objects
     * and parameters of this initializer
     * @param context   Context to manage the initializer
     * @return The newly allocated initializer
     */
    @Override
    public abstract InitializerIface clone(DaalContext context);

    protected native long cGetInput(long cAlgorithm);
}
