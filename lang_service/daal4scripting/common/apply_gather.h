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

#ifndef _APPLY_GATHER_INCLUDED_
#define _APPLY_GATHER_INCLUDED_

#include "cnc4daal.h"
#include "readGraph.h"
#include <cstdlib>

namespace applyGather {

    template< typename Algo > struct context;

    template< int NI >
    struct step1_default {
        template< typename Context >
        int execute( const size_t &, Context & ) const;
    };

    struct step2_default {
        template< typename Context >
        int execute( const size_t &, Context & ) const;
    };

    struct applyTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::tag_tuner<>
    {
        template< typename Arg >
        int compute_on( const size_t tag, Arg & /*arg*/ ) const
        {
            return tag % tuner_base::numProcs();
        }

        int consumed_on( const size_t tag ) const
        {
            return tag % tuner_base::numProcs();
        }

        int get_count( const size_t ) const
        {
            return 1;
        }
    };

    struct gatherTuner : public CnC::step_tuner<>, public CnC::hashmap_tuner, public CnC::preserve_tuner<size_t>
    {
        template< typename Arg >
        int compute_on( const size_t, Arg & /*arg*/ ) const
        {
            return 0;
        }

        int consumed_on( const size_t ) const
        {
            return 0;
        }

        int get_count( const size_t ) const
        {
            return 1;
        }
    };

    template< typename Algo, typename step1=step1_default<1>, typename step2=step2_default >
    class applyGatherContext : public CnC::context< applyGatherContext< Algo, step1, step2 > >
    {
    public:
        typedef Algo algo_type;
        typedef CnC::identityMap< size_t > step1_map;
        typedef CnC::singletonMap< size_t, 0 > step2_map;
        typedef readGraph< applyTuner, applyTuner, (int)Algo::NI > reader_type;

        Algo * algo;
        size_t nBlocks;

        reader_type * reader;

        CnC::dc_step_collection< size_t, size_t, step1_map, step1, applyTuner > step_1;
        CnC::dc_step_collection< size_t, size_t, step2_map, step2, gatherTuner > step_2;

        std::vector< CnC::item_collection< size_t, typename Algo::iomstep1Local_type::input1_type,  applyTuner > * > step1Input;
        CnC::item_collection< size_t, typename Algo::iomstep2Master_type::input1_type, gatherTuner > step2Input;
        CnC::item_collection< size_t, typename Algo::iomstep2Master_type::result_type, gatherTuner > result;

        applyGatherContext(Algo * a = NULL, size_t numBlocks = 0)
            : algo(a),
              nBlocks(numBlocks),
              step_1( *this, "apply" ),
              step_2( *this, "gather" ),
              step1Input(2),
              step2Input( *this, "step2Input"),
              result( *this, "result")
        {
            for(int i = 0; i < (int)Algo::NI; ++i ) {
                step1Input[i] = new CnC::item_collection< size_t, typename Algo::iomstep1Local_type::input1_type,  applyTuner >(*this, "step1Input-" + std::to_string(i));
            }
            for( int i = 0; i < (int)Algo::NI; ++i ) {
                if(i==0) step_1.consumes( *step1Input[i], step1_map() );
                else step_1.consumes( *step1Input[i] );
            }
            step_1.produces( step2Input );
            step_2.consumes( step2Input, step2_map() );
            step_2.produces( result );
            reader = new reader_type(*this, &step1Input);
            if(std::getenv("CNC_DEBUG")) CnC::debug::trace_all(*this);
        }
        ~applyGatherContext()
        {
            for(auto i = step1Input.begin(); i != step1Input.end(); ++i ) delete *i;
            delete reader;
        }
#ifdef _DIST_
        void serialize(CnC::serializer & ser)
        {
            ser & algo & nBlocks;
        }
#endif

    };

    template<>
    template< typename Context >
    int step1_default< 1 >::execute(const size_t & tag, Context & ctxt) const
    {
        typename Context::algo_type::iomstep1Local_type::input1_type pData;
        ctxt.step1Input[0]->get(tag, pData);

        typename Context::algo_type::iomstep1Local_type::result_type res = ctxt.algo->run_step1Local(pData);

        ctxt.step2Input.put(tag, res);
        return 0;
    }

    template<>
    template< typename Context >
    int step1_default< 2 >::execute(const size_t & tag, Context & ctxt) const
    {
        typename Context::algo_type::iomstep1Local_type::input1_type pData1;
        ctxt.step1Input[0]->get(tag, pData1);
        typename Context::algo_type::iomstep1Local_type::input2_type pData2;
        ctxt.step1Input[1]->get(tag, pData2);

        typename Context::algo_type::iomstep1Local_type::result_type res = ctxt.algo->run_step1Local(pData1, pData2);

        ctxt.step2Input.put(tag, res);
        return 0;
    }

    template< typename Context >
    int step2_default::execute(const size_t & tag, Context & ctxt) const
    {
        std::vector< typename Context::algo_type::iomstep2Master_type::input1_type > inp(ctxt.nBlocks);
        for( size_t i = 0; i < ctxt.nBlocks ; i++ ) {
            ctxt.step2Input.get(i, inp[i]);
        }

        typename Context::algo_type::iomstep2Master_type::result_type res = ctxt.algo->run_step2Master(inp);

        ctxt.result.put(tag, res);
        return 0;
    }

    template< typename Algo, typename step1=step1_default< Algo::NI >, typename step2=step2_default >
    struct applyGather
    {
        typedef Algo manager_type;
        typedef applyGatherContext< Algo, step1, step2 > context_type;

        static typename Algo::iomstep2Master_type::result_type
        compute(const TableOrFList & input, Algo & algo)
        {
            size_t nblocks = input.table || input.file.size() ? CnC::tuner_base::numProcs() : (input.flist.size() ? input.flist.size() : input.tlist.size());
            context_type ctxt(&algo, nblocks);
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            if(input.table) {
                ctxt.step1Input[0]->put(CnC::tuner_base::myPid(), input.table);
                ctxt.wait();
            } else if(input.file.size()) {
                ctxt.reader->readInput.put(CnC::tuner_base::myPid(), {input.file});
                ctxt.wait();
            } else {
                assert((input.flist.size() == 0) != (input.tlist.size() == 0));
                for(size_t i = 0; i < input.flist.size(); ++i) {
                    ctxt.reader->readInput.put(i, typename context_type::reader_type::input_type({input.flist[i]}));
                }
                for(size_t i = 0; i < input.tlist.size(); ++i) {
                    ctxt.step1Input[0]->put(i, input.tlist[i]);
                }
                ctxt.wait();
            }
            typename Algo::iomstep2Master_type::result_type res;
            if(CnC::tuner_base::myPid() == 0) ctxt.result.get(0, res);

            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();
            return res;
        }

        static typename Algo::iomstep2Master_type::result_type
        compute(const TableOrFList & input1, const TableOrFList & input2, Algo & algo)
        {
            size_t nblocks = input1.table || input1.file.size() ? CnC::tuner_base::numProcs() : input1.flist.size();
            context_type ctxt(&algo, nblocks);
            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();

            if(input1.table) {
                ctxt.step1Input[0]->put(CnC::tuner_base::myPid(), input1.table);
                ctxt.step1Input[1]->put(CnC::tuner_base::myPid(), input2.table);
                ctxt.wait();
            } else if(input1.file.size()) {
                assert(input2.file.size());
                ctxt.reader->readInput.put(CnC::tuner_base::myPid(), typename context_type::reader_type::input_type({input1.file, input2.file}));
                ctxt.wait();
            } else {
                assert(input1.flist.size() == input2.flist.size());
                assert(input1.tlist.size() == input1.tlist.size());
                assert((input1.flist.size() == 0) != (input1.tlist.size() == 0));
                for(size_t i = 0; i < input1.flist.size(); ++i) {
                    ctxt.reader->readInput.put(i, typename context_type::reader_type::input_type({input1.flist[i], input2.flist[i]}));
                }
                for(size_t i = 0; i < input1.tlist.size(); ++i) {
                    ctxt.step1Input[0]->put(i, input1.tlist[i]);
                    ctxt.step1Input[1]->put(i, input2.tlist[i]);
                }
                ctxt.wait();
            }
            typename Algo::iomstep2Master_type::result_type res;
            if(CnC::tuner_base::myPid() == 0) ctxt.result.get(0, res);

            if(CnC::Internal::distributor::distributed_env()) CnC::Internal::distributor::unsafe_barrier();
            return res;
        }
    };

} // namespace applyGather


#endif // _APPLY_GATHER_INCLUDED_
