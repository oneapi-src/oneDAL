/* file: ForwardResult.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup layers_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the forward layer
 */
public class ForwardResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     *  Constructs the result of the forward layer
     * @param context Context to manage the result of the forward layer
     */
    public ForwardResult(DaalContext context) {
        super(context);
    }

    public ForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the forward layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(ForwardResultId id) {
        if (id == ForwardResultId.value) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the result of the forward layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public KeyValueDataCollection get(ForwardResultLayerDataId id) {
        if (id == ForwardResultLayerDataId.resultForBackward) {
            return (KeyValueDataCollection)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the forward layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(ForwardResultId id, Tensor val) {
        if (id == ForwardResultId.value) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the forward layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(ForwardResultLayerDataId id, KeyValueDataCollection val) {
        if (id == ForwardResultLayerDataId.resultForBackward) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cGetValue(long cObject, int id);

    private native void cSetValue(long cObject, int id, long ntAddr);
}
/** @} */
