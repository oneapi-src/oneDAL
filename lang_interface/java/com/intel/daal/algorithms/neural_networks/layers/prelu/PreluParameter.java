/* file: PreluParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.prelu;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__PRELUPARAMETER"></a>
 * \brief Class that specifies parameters of the prelu layer
 */
public class PreluParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    public PreluParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public PreluParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the index of the dimension for which the weights are applied
     */
    public long getDataDimension() {
        return cGetDataDimension(cObject);
    }

    /**
     *  Sets the index of the dimension for which the weights are applied
     *  @param dataDimension   Starting data dimension index to apply weight
     */
    public void setDataDimension(long dataDimension) {
        cSetDataDimension(cObject, dataDimension);
    }

    /**
    *  Gets the number of weights dimension
    */
    public long getWeightsDimension() {
        return cgetWeightsDimension(cObject);
    }

    /**
     *  Sets the number of weights dimension
     *  @param weightsDimension   The number of weights dimension
     */
    public void setWeightsDimension(long weightsDimension) {
        csetWeightsDimension(cObject, weightsDimension);
    }

    private native long    cInit();
    private native long cGetDataDimension(long cParameter);
    private native void cSetDataDimension(long cParameter, long dataDimension);
    private native long cgetWeightsDimension(long cParameter);
    private native void csetWeightsDimension(long cParameter, long weightsDimension);
}
