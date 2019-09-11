/* file: BatchBase.java */
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
 * @defgroup distributions Distributions
 * @brief Contains classes for distributions
 * @ingroup analysis
 * @{
 */
/**
 * @brief Contains classes for the distributions
 */
package com.intel.daal.algorithms.distributions;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BATCHBASE"></a>
 * @brief Class representing distributions
 *
 * @par References
 *      - Input class
 */
public abstract class BatchBase extends com.intel.daal.algorithms.AnalysisBatch {
    public Input input;     /*!< %Input of the distribution */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs distribution algorithm
     * @param context Context to manage the distribution
     */
    public BatchBase(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated distribution with a copy of input objects
     * and parameters of this distribution
     * @param context   Context to manage the distribution
     * @return The newly allocated distribution
     */
    @Override
    public abstract BatchBase clone(DaalContext context);

    protected native long cGetInput(long cAlgorithm);
}
/** @} */
