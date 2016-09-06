/* file: MultiClassConfusionMatrixParameter.java */
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

/**
 * @brief Contains classes for computing the multi-class confusion matrix*/
package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSCONFUSIONMATRIXPARAMETER"></a>
 * @brief Base class for the parameters of the multi-class confusion matrix algorithm
 */
public class MultiClassConfusionMatrixParameter extends com.intel.daal.algorithms.Parameter {

    public MultiClassConfusionMatrixParameter(DaalContext context) {
        super(context);
    }

    public MultiClassConfusionMatrixParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Sets the parameter of the F-score quality metric
     *  @param beta  Parameter of the F-score
     */
    public void setBeta(double beta) {
        cSetBeta(this.cObject, beta);
    }

    /**
     *  Gets the beta parameter of the F-score quality metric
     *  @return  Parameter of the F-score
     */
    public double getBeta() {
        return cGetBeta(this.cObject);
    }

    /**
     *  Sets the number of classes
     *  @param nClasses  Number of classes
     */
    public void setNClasses(long nClasses) {
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

    private native void cSetBeta(long parAddr, double beta);

    private native double cGetBeta(long parAddr);
}
