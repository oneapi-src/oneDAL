/* file: InitializerIface.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @defgroup initializers Initializers
 * @brief Contains classes for neural network weights and biases initializers
 * @ingroup neural_networks
 * @{
 */
/**
 * @brief Contains classes for the neural network weights and biases initializers
 */
package com.intel.daal.algorithms.neural_networks.initializers;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
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
/** @} */
