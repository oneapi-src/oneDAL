/* file: ConcatParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.concat;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__CONCATPARAMETER"></a>
 * \brief Class that specifies parameters of the concat layer
 */
public class ConcatParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     *  Constructs the parameters for the concat layer
     */
    public ConcatParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public ConcatParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the index of dimension along which concatenation is implemented
     */
    public long getConcatDimension() {
        return cGetConcatDimension(cObject);
    }

    /**
     *  Sets the index of dimension along which concatenation is implemented
     *  @param concatDimension ConcatIndex of dimension along which concatenation is implemented
     */
    public void setConcatDimension(long concatDimension) {
       cSetConcatDimension(cObject, concatDimension);
    }

    private native long cInit();
    private native long cGetConcatDimension(long cParameter);
    private native void cSetConcatDimension(long cParameter, long concatDimension);
}
