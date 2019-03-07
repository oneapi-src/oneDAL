/* file: Result.java */
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
 * @ingroup initializers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.initializers;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__RESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method
 *        of the neural network weights and biases initializer
 */
public class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Result(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the initializer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(ResultId id) {
        if (id == ResultId.value) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(getCObject(), id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the initializer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(ResultId id, Tensor val) {
        if (id == ResultId.value) {
            cSetValue(getCObject(), id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cGetValue(long cObject, int id);
    private native void cSetValue(long cObject, int id, long ntAddr);
}
/** @} */
