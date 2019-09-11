/* file: QualityMetricSetParameter.java */
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
 * @ingroup pca_quality_metric_set
 * @{
 */
package com.intel.daal.algorithms.pca.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC_SET__QUALITYMETRICSETPARAMETER"></a>
 * @brief Parameters for the quality metrics set computation for PCA algorithm
 */
public class QualityMetricSetParameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public QualityMetricSetParameter(DaalContext context, long cParameter, long nComponents, long nFeatures) {
        super(context, cParameter);
        cSetNComponents(this.cObject, nComponents);
        cSetNFeatures(this.cObject, nFeatures);
    }

    /**
     * Sets the number of principal components to compute metrics for
     * @param nComponents Number of principal components to compute metrics for
     */
    public void setNComponents(long nComponents) {
        cSetNComponents(cObject, nComponents);
    }

    /**
     * Gets the number of principal components to compute metrics for
     * @return Number of principal components to compute metrics for
     */
    public long getNComponents(long nComponents) {
        return cGetNComponents(cObject);
    }

    /**
     * Sets the number of features in dataset used as input in PCA
     * @param nFeatures Number of features in dataset used as input in PCA
     */
    public void setNFeatures(long nFeatures) {
        cSetNFeatures(cObject, nFeatures);
    }

    /**
     * Gets the number of features in dataset used as input in PCA
     * @return Number of features in dataset used as input in PCA
     */
    public long getNFeatures(long nFeatures) {
        return cGetNFeatures(cObject);
    }

    private native void cSetNComponents(long cObject, long nComponents);
    private native long cGetNComponents(long cObject);

    private native void cSetNFeatures(long cObject, long nFeatures);
    private native long cGetNFeatures(long cObject);
}
/** @} */
