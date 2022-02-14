/* file: DaalContext.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
