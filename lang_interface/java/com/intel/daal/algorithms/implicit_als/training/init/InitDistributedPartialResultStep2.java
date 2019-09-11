/* file: InitDistributedPartialResultStep2.java */
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
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITDISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * @brief Provides methods to access partial results obtained with the compute() method
 *        of the implicit ALS initialization algorithm
 */
public final class InitDistributedPartialResultStep2 extends InitPartialResultBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of the implicit ALS training algorithm from the context
     * @param context Context to manage the memory in the native part of the partial result object
     */
    public InitDistributedPartialResultStep2(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    InitDistributedPartialResultStep2(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Gets a partial result of the implicit ALS initialization algorithm
     * @param  id   Identifier of the input object, @ref InitDistributedPartialResultStep2Id
     * @return      Partial result that corresponds to the given identifier
     */
    public NumericTable get(InitDistributedPartialResultStep2Id id) {
        if (id != InitDistributedPartialResultStep2Id.transposedData) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
    }

    /**
     * Sets a partial result of the implicit ALS initialization algorithm
     * @param id     Identifier of the input object
     * @param value  Value of the input object
     */
    public void set(InitDistributedPartialResultStep2Id id, NumericTable value) {
        if (id != InitDistributedPartialResultStep2Id.transposedData) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetNumericTable(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetNumericTable(long cObject, int id);
    private native void cSetNumericTable(long cObject, int id, long cNumericTable);
}
/** @} */
