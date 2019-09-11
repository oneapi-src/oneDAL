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
 * @ingroup mse
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.mse;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__PARAMETER"></a>
 * @brief Parameters of the MSE algorithm
 */
public class Parameter extends com.intel.daal.algorithms.optimization_solver.sum_of_functions.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter for MSE algorithm
     * @param context       Context to manage the MSE algorithm
     * @param cParameter    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }
}
/** @} */
