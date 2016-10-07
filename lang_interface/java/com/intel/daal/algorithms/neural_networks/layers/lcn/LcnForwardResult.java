/* file: LcnForwardResult.java */
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

package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method of the forward local contrast normalization layer
 */
public final class LcnForwardResult extends com.intel.daal.algorithms.neural_networks.layers.ForwardResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the forward local contrast normalization layer result
     * @param context   Context to manage the forward local contrast normalization layer result
     */
    public LcnForwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
    * Constructs the forward local contrast normalization layer result
    * @param context   Context to manage the forward local contrast normalization layer result
    * @param cObject   Address of C++ forward result
    */
    public LcnForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the forward local contrast normalization layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(LcnLayerDataId id) {
        if (id == LcnLayerDataId.auxCenteredData || id == LcnLayerDataId.auxSigma || id == LcnLayerDataId.auxC || id == LcnLayerDataId.auxInvMax) {
            return new HomogenTensor(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the forward local contrast normalization layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(LcnLayerDataId id, Tensor val) {
        if (id == LcnLayerDataId.auxCenteredData || id == LcnLayerDataId.auxSigma || id == LcnLayerDataId.auxC || id == LcnLayerDataId.auxInvMax) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();
    private native long cGetValue(long cObject, int id);
    private native void cSetValue(long cObject, int id, long ntAddr);
}
