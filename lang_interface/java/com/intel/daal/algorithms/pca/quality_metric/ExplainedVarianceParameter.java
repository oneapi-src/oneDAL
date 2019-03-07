/* file: ExplainedVarianceParameter.java */
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
