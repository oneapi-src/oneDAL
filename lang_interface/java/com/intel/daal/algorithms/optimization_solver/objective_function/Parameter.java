/* file: Parameter.java */
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
 * @ingroup objective_function
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.objective_function;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__PARAMETER"></a>
 * @brief Parameters of the Objective function algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter for Objective function algorithm
     * @param context       Context to manage the Objective function algorithm
     * @param cParameter    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Constructs the parameter for the Objective function algorithm
     * @param context       Context to manage the parameter for the Objective function algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
     * Sets the 64 bit integer flag that indicates the results to compute
     * @param resultsToCompute The 64 bit integer flag that indicates the results to compute
     */
    public void setResultsToCompute(long resultsToCompute) {
        cSetResultsToCompute(this.cObject, resultsToCompute);
    }

    /**
     * Gets the 64 bit integer flag that indicates the results to compute
     * @return The 64 bit integer flag that indicates the results to compute
     */
    public long getResultsToCompute() {
        return cGetResultsToCompute(this.cObject);
    }

    private native void cSetResultsToCompute(long parAddr, long resultsToCompute);
    private native long cGetResultsToCompute(long parAddr);
}
/** @} */
