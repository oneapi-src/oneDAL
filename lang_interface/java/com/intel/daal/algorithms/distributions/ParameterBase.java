/* file: ParameterBase.java */
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
 * @ingroup distributions
 * @{
 */
package com.intel.daal.algorithms.distributions;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__PARAMETERBASE"></a>
 * @brief Class that specifies parameters of the distribution
 */
public class ParameterBase extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public ParameterBase(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the engine to be used by the distribution
     * @param engine to be used by the distribution
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    private native void cSetEngine(long cObject, long cEngineObject);
}
/** @} */
