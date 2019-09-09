/* file: ReshapeParameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @ingroup reshape
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.reshape;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RESHAPE__RESHAPEPARAMETER"></a>
 * \brief Class that specifies parameters of the reshape layer
 */
public class ReshapeParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     * Constructs the parameter of the reshape layer
     * @param context Context to manage the parameter of the reshape layer
     */
    public ReshapeParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }
    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage the parameter
     * @param cObject Address of C++ parameter
     */
    public ReshapeParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Gets the reshape dimensions
     * @return Array representing the reshape dimensions
     */
    public long[] getReshapeDimensions() {
        return cGetReshapeDimensions(cObject);
    }

    /**
     * Sets the array representing the reshape dimensions
     * @param dims   The array representing the reshape dimensions
     */
    public void setReshapeDimensions(long[] dims) {
        cSetReshapeDimensions(cObject, dims);
    }

    private native long cInit();
    private native void cSetReshapeDimensions(long cObject, long[] dims);
    private native long[] cGetReshapeDimensions(long cObject);
}
/** @} */
