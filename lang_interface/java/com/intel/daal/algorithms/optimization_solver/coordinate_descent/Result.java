/* file: Result.java */
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
 * @ingroup coordinate_descent
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.coordinate_descent;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.OptionalArgument;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__COORDINATE_DESCENT__RESULT"></a>
 * @brief Provides methods to access the results obtained with the compute() method of the
 *        Coordinate Descent algorithm in the batch processing mode
 */
public class Result extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result for the Coordinate Descent algorithm
     * @param context Context to manage the result for the Coordinate Descent algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
    * Constructs the result for the Coordinate Descent algorithm
    * @param context       Context to manage the Coordinate Descent algorithm result
    * @param cResult       Pointer to C++ implementation of the result
    */
    public Result(DaalContext context, long cResult) {
        super(context, cResult);
    }

    private native long cNewResult();
}
/** @} */
