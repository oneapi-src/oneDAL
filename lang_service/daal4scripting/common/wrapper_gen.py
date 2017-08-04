#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

import jinja2
from collections import OrderedDict

# generates typemap function
# requires {{enum_gets}}    list of triplets of members accessed via get(ns, name, type)
#          {{named_gets}}   list of pairs(name, type) of members accessed via type var = getName()
#          {{class_type}}   C++ class
#          {{native_name}}  name of native class (can be None)
typemap_wrapper_template = """
NTYPE native_type({{class_type}} & obj_, int & gc)
{
    if(!obj_.get()) return NNULL;
    const char *names[] = { {% for m in enum_gets+named_gets %} "{{m[1]}}",{% endfor %} "__daalptr__", "" }; /* null string terminated */
    MK_LIST(res_, names, {{('"'+native_name+'"') if False else 'NULL'}}, gc); /* list */
{% for m in enum_gets %}
    {{m[2]}} tmp{{m[1]}} = (obj_)->get({{m[0]}}::{{m[1]}});
    SET_ELT(res_, {{loop.index0}}, native_type(tmp{{m[1]}}, gc), names);
{% endfor %}
{% for m in named_gets %}
    {{m[0]}} tmp{{m[1]}} = obj_->get{{m[1]}}();
    SET_ELT(res_, {{loop.index0}}, native_type(tmp{{m[1]}}, gc), names);
{% endfor %}
    MK_DAALPTR(dp_, new {{class_type}}(obj_), {{class_type}}, gc);
    SET_ELT(res_, {{enum_gets|length+named_gets|length}}, dp_, names);

    return res_;
}
"""

# macro generating class and typemap for DAAL interface classes
# accepts interface name and C++ type
gen_iface_macro = """
{% macro gen_iface(iface_name, iface_type) %}
%shared_ptr({{iface_name}}_i);
%ignore *::get_{{iface_name}}Ptr;
%inline %{
class {{iface_name}}_i : public algo_manager_i
{
public:
    typedef {{iface_type}} {{iface_name}}Ptr_type;
    virtual {{iface_name}}Ptr_type get_{{iface_name}}Ptr()
    {
        return {{iface_name}}Ptr_type();
    }
};
%}
%typemap(in)
(const {{iface_type}})
{
    void *argp = 0;
    int newmem = 0;
    int res = SWIG_ConvertPtrAndOwn($input, &argp, SWIGTYPE_p_daal__services__SharedPtrT_{{iface_name}}_i_t, %convertptr_flags, &newmem);
    if (!SWIG_IsOK(res)) {
        %argument_fail(res, "$type", $symname, $argnum);
    }
    if(argp) {
        daal::services::SharedPtr< {{iface_name}}_i > tmp_kim(*(%reinterpret_cast(argp, daal::services::SharedPtr< {{iface_name}}_i >*)));
        if (newmem & SWIG_CAST_NEW_MEMORY) delete %reinterpret_cast(argp, daal::services::SharedPtr< {{iface_name}}_i >*);
        $1 = tmp_kim->get_{{iface_name}}Ptr();
    } else {
        $1.reset();
    }
}
{% endmacro %}
"""

gen_typedefs_macro = """
{% macro gen_typedefs(ns, template_decl, template_args, mode="Batch", suffix="b", step_spec=None) %}
{% set disttarg = (step_spec.name + ', ') if step_spec.name else "" %}
{% if template_decl|length > 0  %}
    typedef {{ns}}::{{mode}}<{{disttarg + ', '.join(template_args)}}> algo{{suffix}}_type;
{% else %}
    typedef {{ns}}::{{mode}} algo{{suffix}}_type;
{% endif %}
{% if step_spec %}
    typedef {{step_spec.iomanager}}< algo{{suffix}}_type, {{', '.join(step_spec.input)}}, {{step_spec.output}}{{(","+",".join(step_spec.iomargs)) if step_spec.iomargs else ""}} > iom{{suffix}}_type;
{% else %}
    typedef IOManager< algo{{suffix}}_type, services::SharedPtr< typename algo{{suffix}}_type::input_type >, services::SharedPtr< typename algo{{suffix}}_type::result_type > > iom{{suffix}}_type;
{% endif %}
{%- endmacro %}
"""

gen_inst_algo = """
{% macro gen_inst(ns, params_req, params_opt, suffix="", step_spec=None, sp=False) %}
{% if step_spec.construct %}
{% set ctor = '(' + step_spec.construct + ')' %}
{% elif params_req|length > 0  %}
{% set ctor = '(' + ', '.join(params_req.values()).replace('p_', '_p_') + ')' %}
{% else %}
{% set ctor = '' %}
{% endif %}
{% if sp %}
services::SharedPtr< algo{{suffix}}_type > algo{{suffix}}(new algo{{suffix}}_type{{ctor}});
{% else %}
algo{{suffix}}_type algo{{suffix + ctor}};
{% endif %}
{% if params_opt|length %}
        init_parameters(algo{{suffix}}{{'->' if sp else '.'}}parameter);
{% endif %}
{%- endmacro %}
"""

gen_compute_macro = gen_inst_algo + """
{% macro gen_compute(ns, input_args, params_req, params_opt, suffix="", step_spec=None, tonative=True, iomtype=None) %}
{% set iom = iomtype if iomtype else "iom"+suffix+"_type" %}
{% if step_spec %}
{% if step_spec.addinput %}
(const std::vector< typename {{iom}}::input1_type > & input{{', ' + step_spec.extrainput if step_spec.extrainput else ''}})
    {
        {{gen_inst(ns, params_req, params_opt, suffix, step_spec)}}
        int i = 0;
        for(auto data = input.begin(); data != input.end(); ++data, ++i) {
            algo{{suffix}}.input.add({{step_spec.addinput}}, *data);
        }
{% else %}
({% for ia in step_spec.input %}const typename {{iom}}::input{{loop.index}}_type & input{{loop.index}}{{'' if loop.last else ', '}}{% endfor %}{{', ' + step_spec.extrainput if step_spec.extrainput else ''}})
    {
        {{gen_inst(ns, params_req, params_opt, suffix, step_spec)}}
{% for ia in step_spec.input %}
        if(input{{loop.index}}) algo{{suffix}}.input.set({{step_spec.setinput[loop.index0]}}, input{{loop.index}});
{% endfor %}
{% endif %}
{% if step_spec.staticinput %}{% for ia in step_spec.staticinput %}
        if(! use_default(_{{ia[1]}})) algo{{suffix}}.input.set({{ia[0]}},_{{ia[1]}});
{% endfor %}{% endif %}
{% else %}
()
    {
        {{gen_inst(ns, params_req, params_opt, suffix, step_spec)}}
{% for ia in input_args %}
{% if "TableOrFList" in ia[2] %}
        if(!_{{ia[1]}}.table && _{{ia[1]}}.file.size()) _{{ia[1]}}.table = readCSV(_{{ia[1]}}.file);
        if(_{{ia[1]}}.table) algo{{suffix}}.input.set({{ia[0]}}, _{{ia[1]}}.table);
{% else %}
        if(_{{ia[1]}}) algo{{suffix}}.input.set({{ia[0]}}, _{{ia[1]}});
{% endif %}
{% endfor %}
{% endif %}

        algo{{suffix}}.compute();
{% if step_spec %}
        if({{iom}}::needsFini()) {
            algo{{suffix}}.finalizeCompute();
        }
{% endif %}
{% if tonative %}
        auto daalres = {{iom}}::getResult(algo{{suffix}});
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
{% else %}
        return {{iom}}::getResult(algo{{suffix}});
{% endif %}
    }
{%- endmacro %}
"""

# generates wrapper for managing distributed and batch modes
# requires {{ns}}:            namespace
#          {{algo}}:          algo name
#          {{args_decl}}:     non-template arguments for wrapper function
#          {{args_call}}:     non-template arguments to pass on
#          {{input_args}}:    args for setting input
#          {{template_decl}}: template parameters for declaration
#          {{template_args}}: template arguments mapping to their possible values
#          {{params_req}}:    dict of required parameters and their values
#          {{params_opt}}:    dict of parameters and their values
#          {{step_specs}}:     distributed spec
#          {{map_result}}:    Result enum id for getting partial result (can be FULLPARTIAL)
#          {{iface}}:         Interface manager
manager_wrapper_template = gen_typedefs_macro + gen_compute_macro + """
{% if template_decl|length > 0  %}
template<{% for x in template_decl %}{{template_decl[x]['template_decl'] + ' ' + x + ('' if loop.last else ', ')}}{% endfor %}>
{% endif %}
struct {{algo}}_manager{% if template_decl|length != template_args|length %}<{{', '.join(template_args)}}>{% endif %} : public {{algo}}_i
{
{{gen_typedefs(ns, template_decl, template_args, mode="Batch")}}
{% for i in args_decl|match('p_') %}    {{i.rsplit('=', 1)[0].replace(' p_', ' _p_').replace('&', '')}};
{% endfor %}
{% for i in args_decl|match('i_') %}    {{i.rsplit('=', 1)[0].replace(' i_', ' _i_').replace('&', '').replace('const ', '')}};
{% endfor %}
    const bool _distributed;

    {{algo}}_manager({{',\n            '.join(args_decl|match('p_') + ['bool distributed = false'])}})
        : {{algo}}_i()
{% for i in args_call|match('p_') %}
        , _{{i}}({{i}})
{% endfor %}
        , _distributed(distributed)
    {}

private:
{% if params_opt|length %}
    void init_parameters(typename algob_type::parameter_type & parameter)
    {
{% for p in params_opt %}
        if(! use_default(_p_{{p}})) parameter.{{p}} = {{params_opt[p].replace('p_', '_p_')}};
{% endfor %}
    }
{% endif %}

{% for ifc in iface if ifc %}
    virtual {{ifc}}_i::{{ifc}}Ptr_type get_{{ifc}}Ptr()
    {
        {{gen_inst(ns, params_req, params_opt, suffix="b", sp=True)}}
        return algob;
    }
{% endfor %}

    NTYPE batch{{gen_compute(ns, input_args, params_req, params_opt, suffix="b", iomtype=iombatch)}}

{% if step_specs %}
    // Distributed computing
public:
{% for i in range(step_specs|length) %}
{{gen_typedefs(ns, template_decl, template_args, mode="Distributed", suffix=step_specs[i].name, step_spec=step_specs[i])}}
{% endfor %}

{% for i in range(step_specs|length) %}
{% set sname = "run_"+step_specs[i].name %}
    typename iom{{step_specs[i].name}}_type::result_type {{sname + gen_compute(ns, input_args, params_req, params_opt, suffix=step_specs[i].name, step_spec=step_specs[i], tonative=False)}}

{% endfor %}

    enum {NI = {{step_specs[0].inputnames|length}}};

private:
    NTYPE distributed()
    {
        typename iom{{step_specs[-1].name}}_type::result_type daalres = {{pattern}}::{{pattern}}< {{algo}}_manager< {{', '.join(template_args)}} > >::compute(_{{', _'.join(step_specs[0].inputnames)}}, *this);
        int gc = 0;
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
    }

public:
#ifdef _DIST_
{% if params_req|length %}
    {{algo}}_manager() :
{% for i in args_call %}        _{{i}}(){{'' if loop.last else ',\n'}}{% endfor %}
        , _distributed(true)
    {}
{% endif %}

    void serialize(CnC::serializer & ser)
    {
        ser
{% for i in args_call if i != 'i_data' %}            & _{{i.rsplit('=', 1)[0].replace(' p_', ' _p_').replace(' i_', ' _i_')}}
{% endfor %};
    }
#endif

{% endif %}
public:
    NTYPE compute({{',\n                  '.join(args_decl|match('i_'))}})
    {
{% for i in args_call|match('i_') %}        _{{i}} = {{i}};
{% endfor %}

        return {{'_distributed ? distributed() : batch();' if dist else 'batch();'}}
    }
};
{% if step_specs %}
#ifdef _DIST_
namespace CnC {
{% if template_decl|length > 0  %}
template<{% for x in template_decl %}{{template_decl[x]['template_decl'] + ' ' + x + ('' if loop.last else ', ')}}{% endfor %}>
{% endif %}
    static inline void serialize(serializer & ser, {{algo}}_manager{% if template_args|length %}<{{', '.join(template_args)}}>{% endif %} *& t)
    {
        ser & chunk< {{algo}}_manager{% if template_args|length %}<{{', '.join(template_args)}}>{% endif %} >(t, 1);
    }
}
#endif
{% endif %}
"""


# generates HLAPI parent class wrapper (provides compute for all its template specializations)
# requires {{ns}}:            namespace
#          {{algo}}:          algo name
#          {{args_decl}}:     non-template arguments for wrapper function
#          {{iface}}:         Interface manager
parent_wrapper_template = """
class {{algo}}_i : public {{iface[0] if iface[0] else 'algo_manager'}}_i
{
public:
    virtual NTYPE compute({{',\n                          '.join(args_decl|match('i_'))}}) = 0;
};
"""

# generates HLAPI wrapper for one algorithm
# requires {{ns}}:            namespace
#          {{algo}}:          algo name
#          {{args_decl}}:     non-template arguments for wrapper function
#          {{template_decl}}: template arguments spec (name -> (type, values, default))
#          {{dist}}:          boolean: dist mode exists for this algo or not
algo_wrapper_template = """
{% macro tfactory(tmpl_spec, prefix, callargs, dist=False, args=[], indent=4) %}
{{" "*indent}}if( false ) {;}
{% for a in tmpl_spec[0][1]['values'] %}
{% if tmpl_spec[0][1]['values']|length > 1 %}
{{" "*indent}}else if(t_{{tmpl_spec[0][0]}} == "{{a.rsplit('::',1)[-1]}}") {
{% else %}
{{" "*indent}}else {
{% endif %}
{% if tmpl_spec|length == 1 %}
{% set algo_type = prefix + '<' + ', '.join(args+[a]) + ' >' %}
{{" "*(indent+4)}}return services::SharedPtr< {{algo}}_i >(new {{algo_type}}({{', '.join(callargs|match('p_') + ['distributed'])}}));
{{" "*(indent)}}}
{% else %}
{{tfactory(tmpl_spec[1:], prefix, callargs, dist, args+[a], indent+4)}}
{{" "*(indent)}}}
{% endif %}
{% endfor %}
{%- endmacro %}

daal::services::SharedPtr< {{algo}}_i > {{algo}}(
{% for a in args_decl if ' i_' not in a and '=' not in a %}
        {{a}},
{% endfor %}
{% for ta in template_decl if not template_decl[ta]['default'] %}
        const std::string & t_{{ta}},
{% endfor %}
{% for ta in template_decl if template_decl[ta]['default'] %}
        const std::string & t_{{ta}} = "{{template_decl[ta]['default'].rsplit('::',1)[-1]}}",
{% endfor %}
{% for a in args_decl if ' i_' not in a and '=' in a %}
        {{a}},
{% endfor %}
        bool distributed = false
    )
{
{% if template_decl %}
{{tfactory(template_decl.items()|list, algo+'_manager', args_call, dist=dist)}}
  throw std::invalid_argument("no equivalent(s) for C++ template argument(s)");
  return services::SharedPtr< {{algo}}_i >();
{% else %}
   return services::SharedPtr< {{algo}}_i >(new {{algo}}_manager({{', '.join(callargs|match('p_'))}}, distributed));
{% endif %}
}
"""

algo_types_template = """
{% macro tfactory(tmpl_spec, prefix, args=[]) %}
{% for a in tmpl_spec[0][1]['values'] %}
{% if tmpl_spec|length == 1 %}
CnC::Internal::factory::subscribe< typename {{pattern}}::{{pattern}}< {{prefix + '<' + ', '.join(args+[a]) + ' > >::context_type'}} >();
{% else %}
{{tfactory(tmpl_spec[1:], prefix, args+[a])}}
{% endif %}
{% endfor %}
{%- endmacro %}

{% if step_specs %}
{% if template_decl %}
{{tfactory(template_decl.items()|list, algo+'_manager')}}
{% else %}
CnC::Internal::factory::subscribe< {{pattern}}::{{pattern}}< {{algo}}_manager >::context_type >();
{% endif %}
{% endif %}
"""

# Create initialization code.
# Requries {{subscriptions}}
init_template = """
%{
typedef CnC::Internal::dist_init init_type;
init_type * initer = NULL;

struct fini
{
    ~fini()
    {
        if(initer) delete initer;
        initer = NULL;
    }
};
static fini _fini;
%}

#define _DIST_
%inline %{
#ifdef _DIST_
extern "C" {

void daalinit(bool spmd = false, int flag = 0)
{
    if(initer) delete initer;
    auto subscriber = [](){
{{subscriptions.rstrip()}}
    };
    initer = new init_type(subscriber, flag, spmd);
}

void daalfini()
{
    if(initer) delete initer;
    initer = NULL;
}

size_t num_procs()
{
    return CnC::tuner_base::numProcs();
}

size_t my_procid()
{
    return CnC::tuner_base::myPid();
}

} // extern "C"
#endif //_DIST_
%}

"""

##################################################################################
##################################################################################
##################################################################################

def match(a, s):
    return [x for x in a if s in x]

jenv = jinja2.Environment(trim_blocks=True)
jenv.filters['match'] = match

class wrapper_gen(object):
    def __init__(self, ac):
        self.algocfg = ac

    def gen_hlargs(self, template_decl, args_decl):
        """
        Generates a list of tuples, one for each HLAPI argument: (name, type, default)
        """
        res = []
        for a in args_decl:
            if '=' not in a:
                tmp = a.strip().rsplit(' ', 1)
                res.append((tmp[1], tmp[0], None))
        for ta in template_decl:
            if not template_decl[ta]['default']:
                res.append(('t_'+ta, 'string', None))
        for ta in template_decl:
            if template_decl[ta]['default']:
                res.append(('t_'+ta, 'string', template_decl[ta]['default']))
        for a in args_decl:
            if '=' in a:
                tmp1 = a.strip().rsplit('=', 1)
                tmp2 = tmp1[0].strip().rsplit(' ', 1)
                res.append((tmp2[1], tmp2[0], tmp1[1]))
        return res

    ##################################################################################
    def gen_wsinglephased(self, ns, algo):
        """
        Handling single-phased algos which are not part of a multi-phased algo
        """
        cfg = self.algocfg[ns + '::' + algo]
        algostr, retstr, parentstr, typesstr = '', '', '', ''
        hlargs = []
        if len(cfg['params']) == 0:
            return (retstr, parentstr, algostr, typesstr, hlargs)

        jparams = cfg['params'].copy()
        jparams['args_decl'] = jparams['iargs_decl'] + jparams['decl_req'] + jparams['decl_opt']
        jparams['args_call'] = jparams['iargs_call'] + jparams['call_req'] + jparams['call_opt']
        tdecl = cfg['sparams']
        for td in tdecl:
            # Last but not least, we need to provide the template parameter specs
            jparams['template_decl'] = td['template_decl']
            jparams['template_args'] = td['template_args']
            jparams['params_req'] = td['params_req']
            jparams['params_opt'] = td['params_opt']
            # Very simple for specializations
            # but how do we pass only the required args to them from the wrapper?
            # we could have the full input list, but that doesn't work for required parameters
            if td['template_args'] != None:
                if 'dist' in cfg:
                    # a wrapper for distributed mode
                    assert len(tdecl) == 1
                    jparams.update(cfg['dist'])
                    jparams['dist'] = True
                t = jenv.from_string(manager_wrapper_template)
                retstr += t.render(**jparams) + '\n'
            else:
                # our base template spec gets no full template class, only a declaration
                retstr += 'template<' + ', '.join([td['template_decl'][x]['template_decl'] + ' ' + x for x in td['template_decl']]) + '> struct ' + jparams['algo'] + '_batch {};\n'
            if not td['pargs'] != None:
                # this is our actual API wrapper, only once per template (covering all its specializations)
                # the parent class
                t = jenv.from_string(parent_wrapper_template)
                parentstr += t.render(**jparams) + '\n'
                # the C function generating specialized classes
                t = jenv.from_string(algo_wrapper_template)
                algostr += t.render(**jparams) + '\n'
                hlargs += self.gen_hlargs(jparams['template_decl'], jparams['args_decl'])

            t = jenv.from_string(algo_types_template)
            typesstr += t.render(**jparams) + '\n'

            return (retstr, parentstr, algostr, typesstr, hlargs)


    ##################################################################################
    def gen_wrapper(self, ns, algo):
        """
        Here we actually generate the wrapper code. Separating this from preparation
        allows us to cross-reference between algos, for example for multi-phased algos.

        We combine the argument (template, input, parameter) information appropriately.
        We take care of the right order and bring them in the right format for our jinja templates.
        We pass them to the templates in a dict jparams, used a globals vars for jinja.
        """
        return self.gen_wsinglephased(ns, algo)


    ##################################################################################
    def gen_modelmaps(self, ns, algo):
        """
        return string from typemap_wrapper_template for given Model.
        uses entries from 'gets' in Model class def to fill 'named_gets'.
        """
        jparams = self.algocfg[ns + '::' + algo]['model_typemap']
        if len(jparams) > 0:
            t = jenv.from_string(typemap_wrapper_template)
            return t.render(**jparams) + '\n'
        return ''


    ##################################################################################
    def gen_resultmaps(self, ns, algo):
        """
        Generates typedefs for Result type of given namespace.
        Uses target language-specific defines/functions
          - native_type: returns native representation of its argument
          - TMGC(n): deals with GC(refcounting for given number of references (R)
          -
        Looks up Return type and then target-language independently creates lists of its content.
        """
        jparams = self.algocfg[ns + '::' + algo]['result_typemap']
        if len(jparams) > 0:
            t = jenv.from_string(typemap_wrapper_template)
            return t.render(**jparams) + '\n'
        return ''


    ##################################################################################
    def gen_resultmaps(self, ns, algo):
        """
        Generates typedefs for Result type of given namespace.
        Uses target language-specific defines/functions
          - native_type: returns native representation of its argument
          - TMGC(n): deals with GC(refcounting for given number of references (R)
          -
        Looks up Return type and then target-language independently creates lists of its content.
        """
        jparams = self.algocfg[ns + '::' + algo]['result_typemap']
        if len(jparams) > 0:
            t = jenv.from_string(typemap_wrapper_template)
            return t.render(**jparams) + '\n'
        return ''

    ##################################################################################
    def gen_init(self, subscriptions):
        """
        return code for initing (CnC).
        Requires list of CnC context subscriptions.
        """
        import re
        jparams = {'subscriptions' : subscriptions}
        if len(jparams) > 0:
            t = jenv.from_string(init_template)
            return re.sub(r'[\n\s]+CnC::I', '\n        CnC::I', t.render(**jparams)) + '\n'
        return ''

    ##################################################################################
    def gen_ifaces(self, ifaces):
        res = ''
        for i in ifaces:
            tstr = gen_iface_macro + '{{gen_iface("' + i + '", "' + ifaces[i] + '")}}\n'
            t = jenv.from_string(tstr)
            res += t.render({}) + '\n'
        return res
