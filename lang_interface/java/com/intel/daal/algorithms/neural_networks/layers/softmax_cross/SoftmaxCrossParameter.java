/* file: SoftmaxCrossParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.softmax_cross;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX_CROSS__SOFTMAXCROSSPARAMETER"></a>
 * \brief Class that specifies parameters of the softmax cross-entropy layer
 */
public class SoftmaxCrossParameter extends com.intel.daal.algorithms.neural_networks.layers.loss.LossParameter {

    /**
     *  Constructs the parameters for the softmax cross-entropy layer
     */
    public SoftmaxCrossParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public SoftmaxCrossParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the value needed to avoid degenerate cases in logarithm computing
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(cObject);
    }

    /**
     *  Sets the value needed to avoid degenerate cases in logarithm computing
     *  @param accuracyThreshold Value needed to avoid degenerate cases in logarithm computing
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(cObject, accuracyThreshold);
    }

    private native long   cInit();
    private native double cGetAccuracyThreshold(long cParameter);
    private native void   cSetAccuracyThreshold(long cParameter, double accuracyThreshold);
}
