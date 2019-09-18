/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/**
 * @defgroup binary_adaboost binary_adaboost Classifier
 * @brief Contains classes for the binary_adaboost classification algorithm
 * @ingroup boosting
 */
/**
 * @brief Contains classes for the binary_adaboost classification algorithm
 */
package com.intel.daal.algorithms.binary_adaboost;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__binary_adaboost__PARAMETER"></a>
 * @brief binary_adaboost algorithm parameters
 */
public class Parameter extends com.intel.daal.algorithms.boosting.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the accuracy of the binary_adaboost training algorithm
     * @param accuracyThreshold Accuracy of the binary_adaboost training algorithm
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Retrieves the accuracy of the binary_adaboost training algorithm
     * @return Accuracy of the binary_adaboost training algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the maximal number of iterations for the binary_adaboost training algorithm
     * @param maxIterations Maximal number of iterations
     */
    public void setMaxIterations(long maxIterations) {
        cSetMaxIterations(this.cObject, maxIterations);
    }

    /**
     * Retrieves the maximal number of iterations for the binary_adaboost training algorithm
     * @return Maximal number of iterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    private native void cSetAccuracyThreshold(long parAddr, double acc);

    private native double cGetAccuracyThreshold(long parAddr);

    private native void cSetMaxIterations(long parAddr, long nIter);

    private native long cGetMaxIterations(long parAddr);
}
/** @} */
