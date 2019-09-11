/* file: InitDistributedStep4LocalPlusPlusInput.java */
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
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP4LOCALPLUSPLUSINPUT"></a>
 * @brief Input objects for computing initial centroids for the K-Means algorithm.
 *        The class represents input objects for computing initial centroids used
 *        with plusPlus and parallelPlus methods only on the 4th step on a local node.
 */
public final class InitDistributedStep4LocalPlusPlusInput extends com.intel.daal.algorithms.kmeans.init.InitInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep4LocalPlusPlusInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets an input object for computing initial centroids for the K-Means algorithm
     * @param id   Identifier of the input object
     * @param val  Value of the input object     */
    public void set(InitDistributedStep4LocalPlusPlusInputId id, NumericTable val) {
        if (id == InitDistributedStep4LocalPlusPlusInputId.inputOfStep4FromStep3) {
            cSetTable(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for computing initial centroids for the K-Means algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(InitDistributedStep4LocalPlusPlusInputId id) {
        if (id == InitDistributedStep4LocalPlusPlusInputId.inputOfStep4FromStep3) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Sets an input object for computing initial centroids for the K-Means algorithm
    * @param id   Identifier of the input object
    * @param val  Object that corresponds to the given identifier
    */
    public void set(InitDistributedLocalPlusPlusInputDataId id, DataCollection val) {
        if (id == InitDistributedLocalPlusPlusInputDataId.internalInput) {
            cSetDataCollection(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Returns an input object for computing initial centroids for the K-Means algorithm
    * @param id Identifier of the input object
    * @return   Input object that corresponds to the given identifier
    */
    public DataCollection get(InitDistributedLocalPlusPlusInputDataId id) {
        if (id != InitDistributedLocalPlusPlusInputDataId.internalInput) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new DataCollection(getContext(), cGetDataCollection(cObject, id.getValue()));
    }

    private native void cSetTable(long inputAddr, int id, long ntAddr);
    private native long cGetTable(long inputAddr, int id);

    private native void cSetDataCollection(long inputAddr, int id, long ntAddr);
    private native long cGetDataCollection(long inputAddr, int id);

}
/** @} */
