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
 * @defgroup pca_distributed Distributed
 * @ingroup pca
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDSTEP1LOCAL"></a>
 * @brief Runs the PCA algorithm in the first step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * @par References
 *      - ComputeStep class
 *      - InputId class
 *      - PartialCorrelationResultID class
 *      - PartialSVDTableResultID class
 *      - PartialSVDCollectionResultID class
 *      - ResultId class
 *      - Input class
 *      - PartialCorrelationResult class
 *      - PartialSVDResult class
 *      - Result class
 */
public class DistributedStep1Local extends Online {

    /**
     * Constructs the PCA algorithm in the first step of the distributed processing mode
     * by copying input objects and parameters of another PCA algorithm
     * @param context   Context to manage the PCA algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep1Local(DaalContext context, DistributedStep1Local other) {
        super(context, other);
    }

    /**
     * Constructs the PCA algorithm in the first step of the distributed processing mode
     * @param context   Context to manage the PCA algorithm
     * @param cls       Data type to use in intermediate computations for the PCA algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public DistributedStep1Local(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context, cls, method);
    }

    /**
     * Returns the newly allocated PCA algorithm in the first step of the distributed processing mode
     * with a copy of input objects and parameters of this PCA algorithm
     * @param context   Context to manage the PCA algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep1Local clone(DaalContext context) {
        return new DistributedStep1Local(context, this);
    }
}
/** @} */
