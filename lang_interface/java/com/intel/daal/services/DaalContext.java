/* file: DaalContext.java */
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
 * @ingroup memory
 * @{
 */
package com.intel.daal.services;

import com.intel.daal.utils.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 *  <a name="DAAL-CLASS-SERVICES__DAALCONTEXT"></a>
 * @brief Provides the context for managment of memory in the native C++ object
 */
public class DaalContext {

    private ConcurrentLinkedQueue<Disposable> queue;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     *Default constructor
     */
    public DaalContext() {
        queue = new ConcurrentLinkedQueue<Disposable>();
    }

    /**
     * Adds Disposable object to the Context
     */
    public void add(Disposable obj) {
        queue.add(obj);
    }

    /**
     * Removes Disposable object from the Context
     */
    public void remove(Disposable obj) {
        queue.remove(obj);
    }

    /**
     * Frees memory from native C++ object registered in the Context
     */
    public void dispose() {
        Disposable obj = null;
        while ((obj = queue.poll()) != null) {
            obj.dispose();
        }
    }

}
/** @} */
