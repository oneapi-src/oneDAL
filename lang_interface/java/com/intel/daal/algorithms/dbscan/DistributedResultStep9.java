/* file: DistributedResultStep9.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDRESULTSTEP9"></a>
 * @brief Provides methods to access results obtained with the compute() method of the
 *        DBSCAN algorithm in the ninth step of the distributed processing mode
 */
public final class DistributedResultStep9 extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a result of the DBSCAN algorithm from the context
     * @param context Context to manage the memory in the native part of the result object
     */
    public DistributedResultStep9(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedResultStep9();
    }

    DistributedResultStep9(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a result of the DBSCAN algorithm obtained in the ninth step of the distributed processing mode
     * @param  id   Identifier of the input object, @ref DistributedResultStep9Id
     * @return      result that corresponds to the given identifier
     */
    public NumericTable get(DistributedResultStep9Id id) {
        if (id != DistributedResultStep9Id.step9NClusters) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
    }

    /**
    * Sets a result of the DBSCAN algorithm obtained in the ninth step of the distributed processing mode
    * @param id     Identifier of the input object
    * @param value  Value of the input object
    */
    public void set(DistributedResultStep9Id id, NumericTable value) {
        if (id != DistributedResultStep9Id.step9NClusters) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetNumericTable(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewDistributedResultStep9();

    private native long cGetNumericTable(long cObject, int id);
    private native void cSetNumericTable(long cObject, int id, long cNumericTable);
}
/** @} */
