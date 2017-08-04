/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#ifndef _READGRAPH_INCLUDED_
#define _READGRAPH_INCLUDED_

template< typename RTuner, typename OTuner, int NI >
struct readGraph : public CnC::graph
{
    typedef std::array< std::string, NI > input_type;
    typedef std::vector< CnC::item_collection< size_t, data_management::NumericTablePtr, OTuner > * > out_colls_type;
    typedef CnC::identityMap< size_t > step0_map;

    struct step0 {
        template< typename Context >
        int execute( const size_t & tag, Context & ctxt) const
        {
            typename Context::input_type fname;
            ctxt.readInput.get(tag, fname);

            for( int i = 0; i < NI; ++i ) {
                (*ctxt.out_colls)[i]->put(tag, readCSV(fname[i]));
            }
            return 0;
        }
    };

    CnC::item_collection< size_t, input_type, RTuner > readInput;
    CnC::dc_step_collection< size_t, size_t, step0_map, step0, RTuner > step_0;
    std::vector< CnC::item_collection< size_t, data_management::NumericTablePtr, RTuner > * > * out_colls;

    template< typename Ctxt >
    readGraph(CnC::context< Ctxt > & ctxt, out_colls_type * oc, const std::string & name = "reader")
        : CnC::graph(ctxt, name),
          out_colls(oc),
          step_0( ctxt, "read", *this ),
          readInput( ctxt, "readInput")
    {
        step_0.consumes( readInput, step0_map() );
        for( int i = 0; i < NI; ++i ) {
            step_0.produces( *(*out_colls)[i] );
        }
    }
};

#endif // _READGRAPH_INCLUDED_
