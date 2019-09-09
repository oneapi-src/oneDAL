/* file: gc_sharedptr.h */
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

/*
//++
//  Implementation garbage collection hooks for numpy arrays passed to Intel(R) DAAL.
//--
*/

#ifndef __GC_SHAREDPTR_INCLUDED__
#define __GC_SHAREDPTR_INCLUDED__

#include <map>
#include <vector>
#include <services/daal_shared_ptr.h>

typedef std::vector< const PyObject * > gco_vec;
typedef std::map< const void*, gco_vec > gc_map;
static gc_map _pa_map;

static void increment_py_ref(const void *tbl, PyObject * obj) {
    if( ! obj ) return;
    SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    Py_INCREF( obj );
    _pa_map[tbl].push_back(obj);
    SWIG_PYTHON_THREAD_END_BLOCK;
}

static void decrement_py_ref(const void * tbl) {
    SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    gc_map::iterator it = _pa_map.find(tbl);
    if (it != _pa_map.end()) {
        for( gco_vec::iterator i = it->second.begin(); i != it->second.end(); ++i ) {
            Py_DECREF(*i);
        }
        _pa_map.erase(it);
    }
    SWIG_PYTHON_THREAD_END_BLOCK;
}

template<typename T>
static void decrement_py_ref_and_delete(const T * tbl) {
    decrement_py_ref(tbl);
    delete tbl;
}

#endif // __GC_SHAREDPTR_INCLUDED__
