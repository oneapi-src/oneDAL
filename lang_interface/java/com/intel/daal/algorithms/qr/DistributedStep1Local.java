/* file: DistributedStep1Local.java */
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
 * @ingroup qr_distributed
 * @{
 */
package com.intel.daal.algorithms.qr;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP1LOCAL"></a>
 * @brief Computes the results of the QR decomposition algorithm on the first step in the distributed processing mode
 * <!-- \n <a href="DAAL-REF-QR-ALGORITHM">QR decomposition algorithm description and usage models</a> -->
 *
 * @par References
 *      - InputId class. Identifiers of input objects for the QR decomposition algorithm
 *      - PartialResultId class. Identifiers of partial results of the QR decomposition algorithm
 *      - ResultId class. Identifiers of the results of the QR decomposition algorithm
 */
public class DistributedStep1Local extends Online {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the QR decomposition algorithm by copying input objects and parameters
     * of another QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep1Local(DaalContext context, DistributedStep1Local other) {
        super(context, other);
    }

    /**
     * Constructs the QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     * @param cls       Data type to use in intermediate computations of the QR decomposition algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public DistributedStep1Local(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context, cls, method);
    }

    /**
     * Runs the QR decomposition algorithm
     * @return  Partial results of the first step of the QR decomposition algorithm in the distributed processing mode
     */
    @Override
    public DistributedStep1LocalPartialResult compute() {
        super.compute();
        return new DistributedStep1LocalPartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes final results of the QR decomposition algorithm
     * @return  Final results of the QR decomposition algorithm
     */
    @Override
    public Result finalizeCompute() {
        return super.finalizeCompute();
    }

    /**
     * Returns the newly allocated QR decomposition algorithm
     * with a copy of input objects and parameters of this QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep1Local clone(DaalContext context) {
        return new DistributedStep1Local(context, this);
    }
}
/** @} */
