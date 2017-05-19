/* file: LogisticCrossForwardResult.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @ingroup logistic_cross_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC_CROSS__LOGISTICCROSSFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method of the forward logistic cross-entropy layer
 */
public final class LogisticCrossForwardResult extends com.intel.daal.algorithms.neural_networks.layers.loss.LossForwardResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the forward logistic cross-entropy layer result
     * @param context   Context to manage the forward logistic cross-entropy layer result
     */
    public LogisticCrossForwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public LogisticCrossForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the forward logistic cross-entropy layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(LogisticCrossLayerDataId id) {
        if (id == LogisticCrossLayerDataId.auxData || id == LogisticCrossLayerDataId.auxGroundTruth) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the forward logistic cross-entropy layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(LogisticCrossLayerDataId id, Tensor val) {
        if (id == LogisticCrossLayerDataId.auxData || id == LogisticCrossLayerDataId.auxGroundTruth) {
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
/** @} */
