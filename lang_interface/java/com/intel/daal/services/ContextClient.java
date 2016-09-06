/* file: ContextClient.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
 * @brief Contains classes that implement service functionality including memory management,
 *        information about environment, and library version information.
 */
package com.intel.daal.services;

/**
 *  <a name="DAAL-CLASS-SERVICES__CONTEXTCLIENT"></a>
 * @brief Class for management by deletion of the memory allocated for the native C++ object
 */
public abstract class ContextClient implements Disposable {
    private DaalContext _ownContext;

    /**
     * Default constructor
     */
    public ContextClient() {
        _ownContext = null;
    }

    /**
     * Constructs Client of Context and initializes it by provided Context
     */
    public ContextClient(DaalContext context) {
        context.add(this);
        _ownContext = context;
    }

    /**
     * Gets Context
     * @return Context
     */
    public DaalContext getContext() {
        return _ownContext;
    }

    /**
     * Changes Context
     */
    public void changeContext(DaalContext context) {
        if (_ownContext != null) {
            _ownContext.remove(this);
        }
        if (context != null) {
            context.add(this);
        }
        _ownContext = context;
    }

    @Override
    public abstract void dispose();
}
