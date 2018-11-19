/* file: FamilyBatchBase.java */
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
 * @defgroup engines Engines
 * @brief Contains classes for engines
 * @ingroup analysis
 * @{
 */
/**
 * @brief Contains classes for the engines
 */
package com.intel.daal.algorithms.engines;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__FAMILYBATCHBASE"></a>
 * @brief Class representing engines
 *
 * @par References
 *      - Input class
 */
public abstract class FamilyBatchBase extends com.intel.daal.algorithms.engines.BatchBase {
    public Input input;     /*!< %Input of the engine */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs engine algorithm
     * @param context Context to manage the engine
     */
    public FamilyBatchBase(DaalContext context) {
        super(context);
    }

    void add(long numberOfStreams)
    {
        cAdd(numberOfStreams);
    }

    long getNumberOfStreams()
    {
        return cGetNumberOfStreams();
    }

    long getMaxNumberOfStreams()
    {
        return cGetMaxNumberOfStreams();
    }

    private native void cAdd(long numberOfStreams);
    private native long cGetNumberOfStreams();
    private native long cGetMaxNumberOfStreams();
}
/** @} */
