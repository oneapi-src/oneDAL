/* file: FullyConnectedParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.fullyconnected;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__FULLYCONNECTEDPARAMETER"></a>
 * \brief Class that specifies parameters of the fully-connected layer
 */
public class FullyConnectedParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    public FullyConnectedParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the number of layer outputs
     */
    public long getNOutputs() {
        return cGetNOutputs(cObject);
    }

    /**
     *  Sets the number of layer outputs
     *  @param nOutputs A number of layer outputs
     */
    public void setNOutputs(long nOutputs) {
        cSetNOutputs(cObject, nOutputs);
    }

    private native long cInit(long nOutputs);
    private native long cGetNOutputs(long cParameter);
    private native void cSetNOutputs(long cParameter, long nOutputs);
}
