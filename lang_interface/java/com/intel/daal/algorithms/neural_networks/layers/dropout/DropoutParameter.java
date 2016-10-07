/* file: DropoutParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.dropout;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__DROPOUT__DROPOUTPARAMETER"></a>
 * \brief Class that specifies parameters of the dropout layer
 */
public class DropoutParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     *  Constructs the parameters for the dropout layer
     */
    public DropoutParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public DropoutParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the probability that any particular element is retained
     */
    public double getRetainRatio() {
        return cGetRetainRatio(cObject);
    }

    /**
     *  Sets the probability that any particular element is retained
     *  @param retainRatio Probability that any particular element is retained
     */
    public void setRetainRatio(double retainRatio) {
        cSetRetainRatio(cObject, retainRatio);
    }

    /**
     *  Gets the seed for mask elements random generation
     */
    public long getSeed() {
        return cGetSeed(cObject);
    }

    /**
     *  Sets the seed for mask elements random generation
     *  @param seed Seed for mask elements random generation
     */
    public void setSeed(long seed) {
       cSetSeed(cObject, seed);
    }

    private native long   cInit();
    private native double cGetRetainRatio(long cParameter);
    private native void   cSetRetainRatio(long cParameter, double retainRatio);
    private native long   cGetSeed(long cParameter);
    private native void   cSetSeed(long cParameter, long seed);
}
