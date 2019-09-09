/* file: ExplainedVarianceParameter.java */
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
 * @ingroup pca_quality_metric_explained_variance
 * @{
 */
/**
 * @brief Contains classes for computing the explained variance metric */
package com.intel.daal.algorithms.pca.quality_metric;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINEDVARIANCEPARAMETER"></a>
 * @brief Base class for the parameters of the algorithm
 */
public class ExplainedVarianceParameter extends com.intel.daal.algorithms.Parameter {

    /**
     * Constructs the parameters of the quality metric algorithm
     * @param context   Context to manage the parameters of the quality metric algorithm
     */
    public ExplainedVarianceParameter(DaalContext context) {
        super(context);
    }

    public ExplainedVarianceParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Gets the number of components
     * @return Number of components
     */
    public double getNComponents() {
        return cGetNComponents(this.cObject);
    }

    /**
     * Sets the number of components
     * @param nComponents Number of components
     */
    public void setNComponents(long nComponents) {
        cSetNComponents(this.cObject, nComponents);
    }

    /**
     * Gets the number of feautres
     * @return Number of features
     */
    public double getNFeatures() {
        return cGetNFeatures(this.cObject);
    }

    /**
     * Sets the number of features
     * @param nFeatures Number of features
     */
    public void setNFeatures(long nFeatures) {
        cSetNFeatures(this.cObject, nFeatures);
    }

    private native void cSetNComponents(long parAddr, long nComponents);
    private native long cGetNComponents(long parAddr);

    private native void cSetNFeatures(long parAddr, long nFeatures);
    private native long cGetNFeatures(long parAddr);
}
/** @} */
