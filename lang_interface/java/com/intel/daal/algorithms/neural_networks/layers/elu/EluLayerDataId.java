/* file: EluLayerDataId.java */
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
 * @ingroup elu_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.elu;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELU__ELULAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward ELU layer and results for the forward ELU layer
 */
public final class EluLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public EluLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    public static final EluLayerDataId auxData =
        new EluLayerDataId(cGetAuxDataId()); /*!< Data processed at the forward stage of the layer */

    public static final EluLayerDataId auxIntermediateValue =
        new EluLayerDataId(cGetAuxIntermediateValueId()); /*!< Data processed at the forward stage of the layer */

    private static native int cGetAuxDataId();
    private static native int cGetAuxIntermediateValueId();
}
/** @} */
