#!/usr/bin/python
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

from __future__ import print_function
import jinja2
import glob
import os
import re
import sys
import itertools
from fnmatch import fnmatch

try:
    basestring
except NameError:
    basestring = str

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

unique_names = {}

def astuple(value, nums=None):
    """
    Return arg as a tuple. Accepts tuple or converts single element to tuple.
    If nums != None extracts only elements from value which have their index in nums.
    """
    if isinstance(value, tuple):
        return (value[x] for x in range(len(value)) if x in nums) if nums else value
    else:
        return (value,)

def aslist(value):
    """Return arg as a list. Accepts list or converts single element to list."""
    if isinstance(value, list):
        return value
    else:
        return [value,]

def first_non_default(value):
    """
    Returns first element in value which does not contain "default".
    If value is/has only a single element it is returned in any case.
    Otherwise assumes a list/tuple contains at least one non-default element.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
        for x in value:
            if 'default' not in x:
                return x
        print(value)
        assert False
    return value

def unify(n):
    """
    Create a unique name for given string.
    Result depends on order, so you better keep the result if
    you need to use it at different places.
    """
    n = n.rsplit('__',1)[-1]
    unique = n
    i = 1
    while unique in unique_names:
        unique = n + str(i)
        i = i+1
    unique_names[unique] = True
    return unique

def sortforswig(cfg, flt=None):
    def bla(s):
        if len(cfg[s]) == 1:
            pfx = 'a'
        elif len(cfg[s]) == 2:
            pfx = 'b'
        else:
            pfx = 'c'
        if 'Base' in s or 'Iface' in s:
            pfx += 'a'
        elif 'Template' in s:
            pfx += 'b'
        elif 'Partial' in s:
            pfx += 'c'
        else:
            pfx += 'z'
        return pfx + s

    return sorted([k for k in cfg.keys() if flt==None or k in flt], key=bla)

def trans(a):
    """Translate C++ name into Python name using our pyDAAL convention."""
    typemap = { 'double': 'float64',
                'float':  'float32',
                'int':    'intc',}
    try:
        from string import maketrans
        trantab = maketrans("<>", "__")
    except:
        trantab = str.maketrans("<>", "__")
    return '_'.join([(b if b not in typemap else typemap[b]) for b in a.translate(trantab).split('_')]).replace('::', '.')


def ta2pt(tas, pfx=''):
    if len(tas) == 0:
        return None
    #print >> sys.stderr, '!!!ta2pt ' , tas, ' => ', [(pfx+trans(a)) if len(a) else '' for a in tas]
    return [(pfx+trans(a)).replace(' ', '_').rstrip('_') if len(a) else '' for a in tas]

def upperfirst(s):
    return '_'.join([ (x[0].upper() + x[1:]) if len(x) else x for x in s.split('_') ])

def titlelize(v):
    #print >> sys.stderr, '!!!titlelize ' , ' '.join([a.split('.')[-1] for a in v]), ' => ', ''.join([upperfirst(a.split('.')[-1]) for a in v])
    return ''.join([upperfirst(a.split('.')[-1]) for a in v])

def pyArgs(ArgCombos):
    return [ titlelize(ta2pt(ta)) for ta in ArgCombos]

def getcfg(p, tcfg, ns=None):
    if not isinstance(p, basestring):
        return p
    if p.endswith('methods'):
        methods = [first_non_default(x) for x in tcfg[p]]
        if not ns:
            return methods
        else:
            return [ta if '::' in ta else '::'.join(['daal', ns, ta]) for ta in methods]
    elif p == 'steps':
        return tcfg[p]  # if not a else ['::'.join(['daal', p]) for p in tcfg[p]]
    else:
        return ['oops', p]

def get_all_tmplt_combos(tmpl_vals, cfg, ns, nums=None):
    combos = astuple(tmpl_vals, nums=nums)
    r = []
    for i in combos:
        r += [x for x in itertools.product(*[getcfg(b[1], cfg, ns) for b in i])]
    return r

def argvaltuple(val, cfg, sep='='):
    combos = astuple(cfg)
    r = []
    for i in combos:
        r += [sep.join(y) for y in izip((x[0] for x in i), ta2pt(val))]
    return r

def mkinstrule(v, src, dst, algs, copy_so):
    from os.path import join as jp
    s = v + '__'
    vv = v.split('__')
    pyname = jp(vv[-1] + '.py')
    if v == 'daal' or any(e.startswith(s) for e in algs):
        dir = jp(dst, '' if v == 'daal' else 'daal', *vv)
        tname = '__init__.py'
    else:
        dir = jp(dst, 'daal', *vv[:-1])
        tname = pyname
    res = '\n'.join(['\t-mkdir -p ' + dir,
                     '\t((cat license_header.tmpl | sed "/\(\/\*\|\*\/\)/d" | sed s/\*/#/g) && tail -n +9 ' + jp(src, v, pyname) + ') > ' + jp(dir, tname)])
    if copy_so:
        return res + '\n\tcp ' + jp(src, v, '_' + v + '_.so') + ' ' + jp(dir, '$(LIB_' + v + ').so')
    else:
        return res

def normalize_ns(v):
    return [m.replace('::', '__') for m in v]

def find_description(a, cfg):
    if 'descript' in cfg:
        return cfg['descript']
    # otherwise extract from native headers
    try:
        from string import maketrans
        trantab = maketrans("*", " ")
    except:
        trantab = str.maketrans("*", " ")
    root = os.path.join(os.getenv('DAALROOT'), 'include')
    ns = a.split('::')
    files = cfg.get('includes', []) + [i[0].split('include/')[1] for i in cfg.get('needpatch', [])]
    # also try to guess filename itself
    mask = os.path.join(*([root] + ns[:2] + ['_'.join(ns[1:] + ['*.h'])]))
    files += glob.glob(mask)
    exp = re.compile(r'[\\@]brief ([^{]+)\nnamespace ' + ns[-1], re.M|re.S)
    for filename in files:
        try:
            with open(os.path.join(root, filename), "r") as f:
                m = exp.search(f.read())
                if m:
                    return m.group(1).translate(trantab).strip().rstrip('/').strip('\n')
        except:
            pass
    print('Documentation Warning: missing description for ' + a + '\n\twhile searching ' + str(files), file=sys.stderr)
    return a

def get_submodules(iface):
    """Receives a template interface file (with the .i.tmpl stripped) and returns a list of all its immediate
    child modules/packages. This is used for the add_auto_imports macro.  Note: Does not recurse past one level.

    Example:
    get_submodules('algorithms__svm')
    >>> ['prediction', 'quality_metric', 'training']
    """

    nesting_level = len(iface.split('__'))
    iface_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    submodule_candidates = [f.replace('.i.tmpl', '') for f in os.listdir(iface_dir) if fnmatch(f, "*{}*".format(iface))]

    submodules = [sc for sc in submodule_candidates if len(sc.split('__')) - 1 == nesting_level]

    return sorted([s.split('__')[-1] for s in submodules])

def get_interface(c, cfg):
    """Prepend the appropriate interface to class 'c' by examining 'cfg'.  Defaults to 'interface1.'

    Example:
    get_interface('SomeClassInIface2', cfg)
    >>> 'interface2::SomeClassInIface2'
    """
    if c.startswith('interface'):
        return c

    if 'namespaces' not in cfg or c not in cfg['namespaces']:
        return 'interface1::' + c
    else:
        return '::'.join([cfg['namespaces'][c], c]) if cfg['namespaces'][c] else c


if __name__ == "__main__":
    import argparse
    argParser = argparse.ArgumentParser(prog="jinja.py",
                                        description="process template files with jinja2",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument('file', help="the file to process")
    argParser.add_argument('-m', '--mode', default='api', choices=['api', 'dox'], help="Process for API generation or documentation.")
    argParser.add_argument('--vars', default=None, help="provide list global vars: --vars='var1=val1,var2=val2..'")
    args = argParser.parse_args()

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath="."), trim_blocks=True, lstrip_blocks=True)
    env.filters['sortforswig'] = sortforswig
    env.filters['ta2pt'] = ta2pt
    env.filters['titlelize'] = titlelize
    env.filters['pyArgs'] = pyArgs
    env.filters['mkinstrule'] = mkinstrule
    env.filters['getcfg'] = getcfg
    env.filters['get_all_tmplt_combos'] = get_all_tmplt_combos
    env.filters['normalize_ns'] = normalize_ns
    env.filters['argvaltuple'] = argvaltuple
    env.filters['astuple'] = astuple
    env.filters['aslist'] = aslist
    env.filters['first_non_default'] = first_non_default
    env.filters['unify'] = unify
    env.filters['find_description'] = find_description
    env.filters['get_submodules'] = get_submodules
    env.filters['get_interface'] = get_interface
    env.globals['ntypes']   = ['double', 'float', 'int'] # numeric data types
    env.globals['fptypes']  = ['double', 'float']        # floating point data types
    env.globals['stypes']   = ['double']                 # summary statistics types
    env.globals['cmodes']   = ['daal::batch', 'daal::online', 'daal::distributed']  # Compute modes
    env.globals['cputypes'] = ['daal::sse2', 'daal::ssse3', 'daal::sse42', 'daal::avx', 'daal::avx2', 'daal::avx512_mic', 'daal::avx512']
    env.globals['OBFUSCATOR'] = '__WORKAROUND_SWIG_BUG__' # Use this to obfuscate a name for SWIG workarounds
    env.globals['MODE'] = args.mode
    if args.vars:
        for v in args.vars.split(','):
            vsplit = v.split('=')
            vname = vsplit[0].strip()
            vval = vsplit[1].strip()
            env.globals[vname] = vval

    templ = env.get_template(args.file)
    out = templ.render()
    print(out)
