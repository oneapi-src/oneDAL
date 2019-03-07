/* file: InitDistributedParameter.java */
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
 * @ingroup implicit_als_init
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITDISTRIBUTEDPARAMETER"></a>
 * @brief Parameters of the implicit ALS initialization algorithm in the distributed compute mode
 */
public class InitDistributedParameter extends InitParameter {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedParameter(DaalContext context, long parAddr) {
        super(context, parAddr);
    }

    public void setPartition(NumericTable partition) {
        cSetPartition(this.cObject, partition.getCObject());
    }

    public NumericTable getPartition(NumericTable partition) {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPartition(this.getCObject()));
    }

    private native void cSetPartition(long algAddr, long ntAddr);

    private native long cGetPartition(long algAddr);
}
/** @} */
