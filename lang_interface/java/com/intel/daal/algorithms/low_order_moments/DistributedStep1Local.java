/* file: DistributedStep1Local.java */
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
 * @defgroup low_order_moments_distributed Distributed
 * @ingroup low_order_moments
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDSTEP1LOCAL"></a>
 * @brief Computes moments of low order in the distributed processing mode on local nodes.
 * <!-- \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a> -->
 *
 * @par References
 *      - ComputeStep class. Step of distributed processing
 *      - InputId class. Identifiers of the input objects for the low order moments algorithm
 *      - PartialResultId class. Identifiers of partial results
 *      - ResultId class. Identifier of final results
 *      - Parameter class
 */
public class DistributedStep1Local extends Online {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs low order moments algorithm by copying input objects
     * of another low order moments algorithm
     * @param context   Context to manage the low order moments algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep1Local(DaalContext context, DistributedStep1Local other) {
        super(context, other);
    }

    /**
     * Constructs the low order moments algorithm
     * @param context   Context to manage the low order moments algorithm
     * @param cls       Data type to use in intermediate computations,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public DistributedStep1Local(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context, cls, method);
    }

    /**
     * Returns the newly allocated low order moments algorithm
     * with a copy of input objects of this algorithm
     * @param context   Context to manage the low order moments algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep1Local clone(DaalContext context) {
        return new DistributedStep1Local(context, this);
    }
}
/** @} */
