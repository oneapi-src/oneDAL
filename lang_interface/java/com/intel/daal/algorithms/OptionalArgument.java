/* file: OptionalArgument.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__OPTIONALARGUMENT"></a>
 *  \brief Class that provides functionality of the Collection container for Serializable objects
 */
public class OptionalArgument extends com.intel.daal.data_management.data.SerializableBase {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public OptionalArgument(DaalContext context, long size, long dummy) {
        super(context);
        this.cObject = cNewOptionalArgument(size);
    }

    public OptionalArgument(DaalContext context, long cOptionalArgument) {
        super(context);
        this.cObject = cOptionalArgument;
        this.serializedCObject = null;
    }

    public SerializableBase get(long idx) {
        return Factory.instance().createObject(getContext(), cGetValue(this.cObject, idx));
    }

    public void set(SerializableBase value, long idx) {
        cSetValue(this.cObject, value.getCObject(), idx);
    }

    private native long cNewOptionalArgument(long size);

    private native long cGetValue(long cOptionalArgumentAddr, long idx);

    private native void cSetValue(long cOptionalArgumentAddr, long cValueAddr, long idx);
}
/** @} */
