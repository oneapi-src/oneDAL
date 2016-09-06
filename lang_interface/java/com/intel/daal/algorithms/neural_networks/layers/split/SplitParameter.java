/* file: SplitParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.split;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__SPLITPARAMETER"></a>
 * \brief Class that specifies parameters of the split layer
 */
public class SplitParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     *  Constructs the parameters for the split layer
     */
    public SplitParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public SplitParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the number of outputs for forward split layer
     */
    public double getNOutputs() {
        return cGetNOutputs(cObject);
    }

    /**
     *  Sets the number of outputs for forward split layer
     *  @param nOutputs Number of outputs for forward split layer
     */
    public void setNOutputs(long nOutputs) {
        cSetNOutputs(cObject, nOutputs);
    }

    /**
     *  Gets the number of inputs for backward split layer
     */
    public long getNInputs() {
        return cGetNInputs(cObject);
    }

    /**
     *  Sets the number of inputs for backward split layer
     *  @param nInputs Number of inputs for backward split layer
     */
    public void setNInputs(long nInputs) {
       cSetNInputs(cObject, nInputs);
    }

    private native long cInit();
    private native long cGetNOutputs(long cParameter);
    private native void cSetNOutputs(long cParameter, long nOutputs);
    private native long cGetNInputs(long cParameter);
    private native void cSetNInputs(long cParameter, long nInputs);
}
