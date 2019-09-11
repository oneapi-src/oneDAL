/* file: InitDistributedStep5MasterPlusPlusPartialResult.java */
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
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP5MASTERPLUSPLUSPARTIALRESULT"></a>
 * @brief Provides methods to access partial results of computing initial centroids for
 *        the K-Means algorithm in the distributed processing mode
 */
public final class InitDistributedStep5MasterPlusPlusPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Default constructor. Constructs empty object
     * @param context       Context to manage the partial result of computing initial centroids for the K-Means algorithm
     */
    public InitDistributedStep5MasterPlusPlusPartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public InitDistributedStep5MasterPlusPlusPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of computing initial centroids for the K-Means algorithm
     * @param id Identifier of the partial result object
     * @return   Partial result object that corresponds to the given identifier
     */
    public NumericTable get(InitDistributedStep5MasterPlusPlusPartialResultId id) {
        if ((id == InitDistributedStep5MasterPlusPlusPartialResultId.candidates) ||
            (id == InitDistributedStep5MasterPlusPlusPartialResultId.weights)) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Sets a partial result object for computing initial centroids for the K-Means algorithm
    * @param id   Identifier of the partial result object
    * @param val  Object that corresponds to the given identifier
    */
    public void set(InitDistributedStep5MasterPlusPlusPartialResultId id, NumericTable val) {
        if ((id == InitDistributedStep5MasterPlusPlusPartialResultId.candidates) ||
            (id == InitDistributedStep5MasterPlusPlusPartialResultId.weights)) {
            cSetTable(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewPartialResult();

    private native void cSetTable(long inputAddr, int id, long ntAddr);
    private native long cGetTable(long inputAddr, int id);

}
/** @} */
