/* file: QualityMetricSetParameter.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @ingroup multi_class_classifier_quality_metric_set
 * @{
 */
/**
 * @brief Contains classes for computing the multi-class confusion matrix
 */
package com.intel.daal.algorithms.multi_class_classifier.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__QUALITY_METRIC_SET__QUALITYMETRICSETPARAMETER"></a>
 * @brief Class for the parameter of the multi-class SVM algorithm
 */

public class QualityMetricSetParameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public QualityMetricSetParameter(DaalContext context, long cParameter, long nClasses) {
        super(context, cParameter);
        cSetNClasses(this.cObject, nClasses);
    }

    /**
     *  Gets the number of classes
     *  @return  Number of classes
     */
    public long getNClasses() {
        return cGetNClasses(this.cObject);
    }

    private native void cSetNClasses(long parAddr, long nClasses);

    private native long cGetNClasses(long parAddr);
}
/** @} */
