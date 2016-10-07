/* file: SoftmaxParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.softmax;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__SOFTMAXPARAMETER"></a>
 * \brief Class that specifies parameters of the softmax layer
 */
public class SoftmaxParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    public SoftmaxParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public SoftmaxParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the index of the dimension to calculate softmax
     */
    public long getDimension() {
        return cGetDimension(cObject);
    }

    /**
     *  Sets the index of the dimension to calculate softmax
     *  @param dimension   SoftmaxIndex of the dimension to calculate softmax
     */
    public void setDimension(long dimension) {
        cSetDimension(cObject, dimension);
    }

    private native long cInit();
    private native long cGetDimension(long cParameter);
    private native void cSetDimension(long cParameter, long dimension);
}
