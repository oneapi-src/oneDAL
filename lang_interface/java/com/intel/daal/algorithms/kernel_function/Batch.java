/* file: Batch.java */
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
 * @defgroup kernel_function Kernel Functions
 * @brief Contains classes for computing kernel functions
 * @ingroup analysis
 * @{
 */
package com.intel.daal.algorithms.kernel_function;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__BATCH"></a>
 * \brief Computes the kernel function in the batch processing mode
 * <!-- \n<a href="DAAL-REF-KERNEL_FUNCTION-ALGORITHM">Kernel function algorithm description and usage models</a> -->
 *
 * \par References
 *      - Parameter class
 */
public abstract class Batch extends AnalysisBatch {
    protected Batch(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated kernel function algorithm with a copy of input objects
     * and parameters of this kernel function algorithm
     * @param context  Context to manage the kernel function
     *
     * @return The newly allocated algorithm
     */
    @Override
    public abstract Batch clone(DaalContext context);
}
/** @} */
