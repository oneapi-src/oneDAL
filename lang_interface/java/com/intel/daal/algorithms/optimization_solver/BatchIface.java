/* file: BatchIface.java */
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
 * @addtogroup optimization_solver
 * @{
 */
/**
 * @brief Contains classes for computing the optimization solvers
 */
package com.intel.daal.algorithms.optimization_solver;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.ComputeMode;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__BATCH"></a>
 * @brief %Base interface for the Optimization solver algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-OPTIMIZATION_SOLVER-ALGORITHM">Optimization solver algorithm description and usage models</a> -->
 */
public abstract class BatchIface extends com.intel.daal.algorithms.AnalysisBatch {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the optimization solver algorithm in the batch processing mode
     * @param context  Context to manage the optimization solver algorithm
     */
    public BatchIface(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated Optimization solver algorithm
     * with a copy of input objects and parameters of this Optimization solver algorithm
     * @param context    Context to manage the Optimization solver algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract BatchIface clone(DaalContext context);
}
/** @} */
