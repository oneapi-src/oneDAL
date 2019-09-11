/* file: DistributedPartialResultStep13.java */
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
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.dbscan;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP13"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the
 *        DBSCAN algorithm in the thirteenth step of the distributed processing mode
 */
public final class DistributedPartialResultStep13 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of the DBSCAN algorithm from the context
     * @param context Context to manage the memory in the native part of the partial result object
     */
    public DistributedPartialResultStep13(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedPartialResultStep13();
    }

    DistributedPartialResultStep13(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of the DBSCAN algorithm obtained in the thirteenth step of the distributed processing mode
     * @param  id   Identifier of the input object, @ref DistributedPartialResultStep13Id
     * @return      partial result that corresponds to the given identifier
     */
    public NumericTable get(DistributedPartialResultStep13Id id) {
        if (id != DistributedPartialResultStep13Id.step13AssignmentQueries) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
    }

    /**
    * Sets a partial result of the DBSCAN algorithm obtained in the thirteenth step of the distributed processing mode
    * @param id     Identifier of the input object
    * @param value  Value of the input object
    */
    public void set(DistributedPartialResultStep13Id id, NumericTable value) {
        if (id != DistributedPartialResultStep13Id.step13AssignmentQueries) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetNumericTable(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewDistributedPartialResultStep13();

    private native long cGetNumericTable(long cObject, int id);
    private native void cSetNumericTable(long cObject, int id, long cNumericTable);
}
/** @} */
