/* file: ParameterDefaultDense.java */
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
 * @ingroup sgd
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sgd;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.algorithms.optimization_solver.sgd.BaseParameter;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETERDEFAULTDENSE"></a>
 * @brief ParameterDefaultDense of the SGD algorithm
 */
public class ParameterDefaultDense extends BaseParameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter for SGD algorithm
     * @param context       Context to manage the parameter for SGD algorithm
     */
    public ParameterDefaultDense(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for SGD algorithm
     * @param context       Context to manage the SGD algorithm
     * @param cParameter    Pointer to C++ implementation of the parameter
     */
    public ParameterDefaultDense(DaalContext context, long cParameter) {
        super(context, cParameter);
    }
}
/** @} */
