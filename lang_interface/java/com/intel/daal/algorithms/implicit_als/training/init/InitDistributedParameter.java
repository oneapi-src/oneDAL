/* file: InitDistributedParameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
