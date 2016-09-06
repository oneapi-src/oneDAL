/* file: DistributedStep1Local.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

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
 * \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a>
 *
 * @par References
 *      - ComputeStep class
 *      - Method class
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
