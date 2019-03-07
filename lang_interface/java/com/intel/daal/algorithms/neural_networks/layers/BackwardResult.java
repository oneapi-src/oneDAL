/* file: BackwardResult.java */
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
 * @ingroup layers_backward
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward layer
 */
public class BackwardResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the backward layer
     * @param context   Context to manage the result of the backward layer
     */
    public BackwardResult(DaalContext context) {
        super(context);
    }

    public BackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the backward layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(BackwardResultId id) {
        if (id == BackwardResultId.gradient || id == BackwardResultId.weightDerivatives || id == BackwardResultId.biasDerivatives) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the result of the backward layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public KeyValueDataCollection get(BackwardResultLayerDataId id) {
        if (id == BackwardResultLayerDataId.resultLayerData) {
            return (KeyValueDataCollection)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the backward layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(BackwardResultId id, Tensor val) {
        if (id == BackwardResultId.gradient || id == BackwardResultId.weightDerivatives || id == BackwardResultId.biasDerivatives) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the backward layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(BackwardResultLayerDataId id, KeyValueDataCollection val) {
        if (id == BackwardResultLayerDataId.resultLayerData) {
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
