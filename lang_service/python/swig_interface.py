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

import os
from pprint import pprint
from os.path import join as jp
from collections import defaultdict
from parse import parse_header
try:
    basestring
except NameError:
    basestring = str

from jinja import first_non_default

###############################################################################
# Managing SWIG interface files (jinja templates).
# - Uses parse.py to read in C++ headers files.
# - reads in an existing config for each namespace and compares it to what it
#   found in the C++ headers
# - Converts resulting data strucutures into SWIG consumable files
#   this requries some recursive searching and expansion
###############################################################################


###############################################################################
def cleanup_ns(fname, ns):
    """return a sanitized namespace name"""
    # strip of namespace 'interface*'
    while len(ns) and ns[-1].startswith('interface'):
        del ns[-1]
    # cleanup duplicates
    while len(ns) >= 2 and ns[-1] == ns[len(ns)-2]:
        del ns[-1]
    # we should now have our namespace hierachy
    if len(ns) == 0 or ns[0] != 'daal':
        print(fname + ":0: Warning: No namespace (starting with daal) found in " + fname + '. Ignored.')
        return False
    nsn = '::'.join(ns[1:])
    # namespace 'daal' is special, it's empty
    if len(nsn) == 0:
        nsn = 'daal'

    return nsn


###############################################################################
###############################################################################
def get_parent(ns):
    tmp = ns.rsplit('::', 1)
    return tmp[0] if len(tmp) > 1 else 'daal'

###############################################################################
###############################################################################
def compare(ns, a, b, allowed_diffs, ignores):
    """
    Helper function comparing two configs.
    Iterating through all keys in config a and checks if the same exists in b.
    Certain fields are skipped.
    A Warning message will be printed for each difference found.
    """
    cfga = a[0]
    cfgb = b[0]
    # go through all entries in a
    for x in cfga:
        if x == 'namespaces':
            compare_interfaces(a, b)
            continue
        if x not in ['package', 'module', 'namespace', 'deps', 'renames', 'computes', 'ignore',]:
            if not cfga[x]:
                continue
            # check if in b
            if x in cfgb:
                # go through all sub-entries
                for y in cfga[x]:
                    found = False
                    # we need to account for our special '!' syntax for jinja processing
                    for yy in ([y, y.strip('!'), y+'!', '!'+y] if isinstance(y, basestring) else [y]):
                        if yy in cfgb[x]:
                            if isinstance(cfga[x],dict) and (not isinstance(cfgb[x],dict) or cfga[x][y] != cfgb[x][yy]):
                                if '::'.join([ns, x, y.strip('!')]) not in allowed_diffs:
                                    # value is not identical and it's not in allowed diffs
                                    print(b[1] + ':0: Warning: ' + x + '->' + y + ': ' + str(cfga[x][y]) + ' differs')
                                    print(a[1] + ':0: Warning: from what is defined here')
                            found = True
                            break
                    if not found and y not in ignores and '::'.join([ns, x, str(y)]) not in allowed_diffs:
                        # key not found in b
                        print(b[1] + ':0: Warning: ' + x + '->' + str(y) + ' not found for ' + ns)
                        print(a[1] + ':0: Warning: but it is defined here')
            else:
                if '::'.join([ns, x]) not in allowed_diffs:
                    # key not found in b and not in allowed diffs
                    print(b[1] + ':0: Warning: ' + x + ' not found for ' + ns)
                    print(a[1] + ':0: Warning: but it is defined here')


def compare_interfaces(a, b):
    fnamea = a[1]
    fnameb = b[1]
    nsa = a[0].get('namespaces', None)
    nsb = b[0].get('namespaces', None)

    for cl, iface in nsa.items():
        if len(cl.split('::')) == 2:
            # 'cl' is a method. Get its class
            cl = cl.split('::')[0]

        nsb_iface = nsb.get(cl, None) if nsb else None
        if nsb_iface != iface:
            print("{}: Warning: {} is using {} here".format(fnamea, cl, iface))
            print("{}: Warning: but is using {} here".format(fnameb, nsb_iface if nsb_iface else 'interface1'))


###############################################################################
class namespace(object):
    """Holds all extracted data of a namespace"""
    def __init__(self, name):
        self.classes = {}
        self.enums = {}
        self.headers = []
        self.includes = set()
        self.name = name
        self.need_methods = False
        self.steps = set()
        self.children = set()
        self.globals = {}
        self.interface = {}


###############################################################################
    def resolve_methods(self, ns, namespace_dict):
        """
        Find and return first list of Method enum in namespace hierachy.
        Getting called assumes the ns needs methods and so will find in parents
        if there are none in the ns itself.
        """
        if 'Method' in self.enums:
            # this ns has its own definition of methods
            pfx = '' if ns == self.name else 'daal::' + self.name + '::'

            rev = defaultdict(list)
            for m in self.enums['Method']:
                rev[self.enums['Method'][m]].append(pfx + m)
            res = sorted(rev[''])
            del rev['']
            for m in rev:
                if m.isdigit():
                    res.append(sorted(rev[m]) if len(rev[m]) > 1 else rev[m][0])
                elif 'default' in m:
                    res.append(sorted(list(rev[m]) + [pfx + m]))
                else:
                    print(rev)
                    assert False
            return list(res)

        # we do not have methods in this namespace
        # let's find them in our parents
        split_ns = self.name.rsplit('::', 1)
        if len(split_ns) > 1 and split_ns[0] in namespace_dict:
            return namespace_dict[split_ns[0]].resolve_methods(ns, namespace_dict)
        else:
            return []


###############################################################################
    def resolve_deps(self, namespace_dict, deps):
        """Performs one recursive pass through dependences from #include diirectives.
           Adds dependent namespaces to its list of deps and return it."""
        # let's determine which namespace have the include files a given ns needs to #define as USE_*
        #   also determine the interface files to be imported
        for inc in self.includes:
            # check which namespace the #included file belongs to
            for ns2 in namespace_dict:
                if inc in namespace_dict[ns2].headers:
                    if ns2 not in ['daal', 'algorithms', 'services', 'data_management', self.name]:
                        # we found the namespace -> add once
                        deps[self.name].add(ns2)
                        # we also add all the deps that
                        for n in deps[ns2]:
                            deps[self.name].add(n)
                    break
        return deps[self.name]


###############################################################################
    def resolve_computes(self, namespace_dict):
        """
        Determine which compute wrappers are needed and return dict.
        The values are strings with the standard jinja macro calls.
        """
        computes = {}
        # Determine the compute() wrappers we need
        #  (e.g. if there are Batch, Online, Distributed algorithms)
        if 'Batch' in self.classes:
            computes['Batch'] = '{{add_compute("Batch")}}'
        if 'Online' in self.classes:
            computes['Online'] = '{{add_compute2(ns, cfg, "Online")}}'
        if 'Distributed' in self.classes:
            computes['Distributed'] = '{{add_compute2(ns, cfg, "Distributed")}}'
        return computes


###############################################################################
    def format_setgets(self):
        """
        Returns a list of strings with SWIG %rename directives, one for each set/get method.
        The new name is generated from the argument types.
        """
        rs = []
        for c in self.classes:
            if len(self.classes[c].setgets) and isinstance(self.classes[c].setgets, (list, tuple)):
                for r in self.classes[c].setgets:
                    iface = self.interface.get(c, 'interface1')
                    rs.append('%rename(' + r[0] + r[2].replace('Id', '') + ') /*' + r[1] + '*/ daal::{{ns}}::' + iface + '::' + c + '::' + r[3])
        return rs


###############################################################################
    def as_iface(self, namespace_dict, deps):
        """returns a dict in the iface-tmpl format for our i.tmpl files"""
        cfg = {
            'classes'   : [c for c in self.classes if not self.classes[c].template_args],
            'computes'  : self.resolve_computes(namespace_dict),
            'deps'      : deps,
            'includes'  : self.headers,
            'module'    : self.name.rsplit('::', 1)[-1],
            'namespace' : self.name,
            'namespaces': self.interface,
            'package'   : self.name.rsplit('::', 1)[0].replace('::', '.'),
            'renames'   : self.format_setgets(),
            'steps'     : ['daal::' + x for x in self.steps],
            'templates' : {c: self.classes[c].template_args for c in self.classes if self.classes[c].template_args and not self.classes[c].partial},
        }

        for c in self.classes:
            cfg['templates'].update({f[0]: f[1] for f in self.classes[c].templates})
        if self.need_methods:
            cfg['methods'] = self.resolve_methods(self.name, namespace_dict)
            if not len(cfg['methods']):
                print(self.name)
                assert False
            # if the methods are from a parent, we assume they are all from the same ns.
            # we need to prepend the parent's fully qualified namespace to each
            # method template parameter.
            if '::' in cfg['methods'][0]:
                tmp = cfg['methods'][0].rsplit('::', 1)[0]
                for t in cfg['templates']:
                    for a in cfg['templates'][t]:
                        if a[0] == 'method' and len(a[2]):
                            a[2] = tmp + '::' + a[2]
        # template default values get resolved to the first non-default-name of the same
        # enum value
        for tmpl in cfg['templates']:
            for tmplarg in cfg['templates'][tmpl]:
                if len(tmplarg) >= 3 and tmplarg[2] in namespace_dict['daal'].globals.keys():
                    # Currently NN modules default to float while everything else is double
                    if 'neural_networks' in cfg['namespace']:
                        tmplarg[2] = 'float'
                    else:
                        tmplarg[2] = namespace_dict['daal'].globals[tmplarg[2]]
                if len(tmplarg) >= 3 and tmplarg[1] in cfg and 'default' in tmplarg[2]:
                    for val in cfg[tmplarg[1]]:
                        if tmplarg[2] == val:
                            break
                        if isinstance(val, (tuple, list)) and tmplarg[2] in val:
                            tmplarg[2] = first_non_default(val)
                            break

        return cfg


###############################################################################
    def from_iface(self, fname):
        if not os.path.isfile(fname):
            return None
        cfgstr = None
        with open(fname, "r") as f:
            for l in f:
                if cfgstr:
                    if '%}' in l:
                        cfgstr += '}'
                        return eval(cfgstr, {'fptypes': 'fptypes', 'cmodes': 'cmodes', 'ntypes': 'ntypes', 'stypes': 'stypes'})
                    else:
                        cfgstr += l
                elif 'set cfg = {' in l:
                    cfgstr = '{'
        return None


###############################################################################
    def is_empty(self):
        return not any(len(x) > 0 for x in [self.classes, self.enums, self.steps])


###############################################################################
    def write(self, cfg, fname):
        print("Writing " + fname)
        with open(fname, 'w') as template_file:
            template_file.write('cfg =\n')
            pprint(cfg, stream=template_file, width=110)


###############################################################################
###############################################################################
class swig_interface(object):
    """collecting and generating data for SWIG"""

    # classes/functions we generally ignore
    ignores = ['AlgorithmContainerIface', 'AnalysisContainerIface',
               'PredictionContainerIface', 'TrainingContainerIface', 'DistributedPredictionContainerIface',
               'BatchContainerIface', 'OnlineContainerIface', 'DistributedContainerIface',
               'BatchContainer', 'OnlineContainer', 'DistributedContainer',
               'serializeImpl', 'deserializeImpl', 'serialImpl',
               'getEpsilonVal', 'getMinVal', 'getMaxVal', 'getPMMLNumType', 'getInternalNumType', 'getIndexNumType',
               'allocateNumericTableImpl', 'allocateImpl',
               'setPartialResultStorage', 'addPartialResultStorage', 'create']

    # file we ignore/skip
    ignore_files = ['daal_shared_ptr.h', 'daal.h', 'daal_win.h', 'algorithm_base_mode_batch.h',
                    'algorithm_base.h', 'algorithm.h', 'daal_kernel_defines.h', 'data_utils.h',
                    'decision_forest_classification_model_builder.h', 'gbt_regression_model_builder.h',
                    'gbt_classification_model_builder.h', 'svm_model_builder.h', 'linear_regression_model_builder.h',
                    'multi_class_classifier_model_builder.h', 'logistic_regression_model_builder.h']

    ignore_dirs = [
        'internal', # don't need to generate swig files for internal directories
        'data_management/features',                  # temporary disabled
        'data_management/data_source/modifiers',     # temporary disabled
        'data_management/data_source/modifiers/csv', # temporary disabled
        'data_management/data_source/modifiers/sql',  # temporary disabled
        'algorithms/dbscan',                                 # new feature, will not be enabled in pydaal
        'algorithms/lasso_regression',                       # new feature, will not be enabled in pydaal
        'algorithms/optimization_solver/coordinate_descent'  # new feature, will not be enabled in pydaal
    ]

    # allowed diffs
    # matched against $namespace::$symbol; $symbol can be be a class, template or key in our config
    allowed_diffs = ['algorithms::implicit_als::training::init::templates::Distributed',
                     'algorithms::implicit_als::training::init::templates::Batch',
                     'algorithms::implicit_als::training::templates::Distributed',
                     'algorithms::implicit_als::training::templates::Batch',
                     'algorithms::multi_class_classifier::prediction::templates::Batch',
                     'algorithms::implicit_als::training::stages',
                     'algorithms::implicit_als::training::dmethods',
                     'algorithms::implicit_als::training::bmethods',
                     'algorithms::implicit_als::training::init::dmethods',
                     'algorithms::implicit_als::training::init::bmethods',
                     'algorithms::pca::templates::PartialResult<daal::algorithms::pca::svdDense>::initialize',
                     'algorithms::pca::templates::PartialResult<daal::algorithms::pca::svdDense>::allocate',
                     'algorithms::pca::templates::PartialResult<daal::algorithms::pca::correlationDense>::initialize',
                     'algorithms::pca::templates::PartialResult<daal::algorithms::pca::correlationDense>::allocate',
                     'algorithms::kmeans::init::templates::Distributed',
                     'algorithms::kmeans::init::s1methods',
                     'algorithms::kmeans::init::s2mmethods',
                     'algorithms::kmeans::init::s2l34methods',
                     'algorithms::kmeans::init::s5methods',
                     'algorithms::low_order_moments::templates::DistributedInput',]

    ignore_ns = ['daal', 'algorithms', 'services', 'data_management', 'data_feature_utils']

###############################################################################
    def __init__(self, include_root):
        self.include_root = include_root
        self.namespace_dict = defaultdict(namespace)


###############################################################################
    def read(self):
        """
        Walk through each directory in the root dir and read in C++ headers.
        Creating a namespace dictionary. Of course, the it needs to go through every header file to find out
        what namespace it is affiliated with. Once it does this, we have a dictionary where the key is the namespace
        and the values are namespace class objects. These objects carry all information as extracted by parse.py.
        """
        for (dirpath, dirnames, filenames) in os.walk(self.include_root):
            if any(dirpath.endswith(x) for x in swig_interface.ignore_dirs):
                continue
            for filename in filenames:
                if filename.endswith('.h') and not 'internal' in dirpath and not any(filename.endswith(x) for x in swig_interface.ignore_files):
                    fname = jp(dirpath,filename)
                    print(fname)
                    with open(fname, "r") as header:
                        parsed_data = parse_header(header, swig_interface.ignores)

                    ns = cleanup_ns(fname, parsed_data['ns'])
                    if ns:
                        if ns not in self.namespace_dict:
                            self.namespace_dict[ns] = namespace(ns)
                        pns = get_parent(ns)
                        if pns not in self.namespace_dict:
                            self.namespace_dict[pns] = namespace(pns)
                        if ns != 'daal':
                            self.namespace_dict[pns].children.add(ns)
                        self.namespace_dict[ns].includes = self.namespace_dict[ns].includes.union(parsed_data['includes'])
                        self.namespace_dict[ns].steps = self.namespace_dict[ns].steps.union(parsed_data['steps'])
                        self.namespace_dict[ns].classes.update(parsed_data['classes'])
                        self.namespace_dict[ns].enums.update(parsed_data['enums'])
                        self.namespace_dict[ns].globals.update(parsed_data['globals'])
                        self.namespace_dict[ns].headers.append(fname.replace(self.include_root, '').lstrip('/'))
                        self.namespace_dict[ns].interface.update(parsed_data['namespaces'])
                        if parsed_data['need_methods']:
                            self.namespace_dict[ns].need_methods = True


###############################################################################
    def digest(self):
        """
        1. Process raw data in our namespace_dict.
        """
        # sort include files to put *_types file first in list
        for ns in self.namespace_dict:
            self.namespace_dict[ns].headers.sort(key=lambda x: x if '_types' not in x else '_')



###############################################################################
    def write_and_compare(self, newdir, olddir):
        """
        Write the extract data into files with extension as newdir/*.i.tmpl.new.
        Then compare with corresponding file in dir "olddir".
        """
        # let's recursively resolve all dependences between namespaces
        all_deps = defaultdict(set)
        added = 1
        while added > 0:
            added = 0
            for ns in self.namespace_dict:
                oldcnt = len(all_deps[ns])
                all_deps[ns] = self.namespace_dict[ns].resolve_deps(self.namespace_dict, all_deps)
                added += len(all_deps[ns]) - oldcnt
        for ns in self.namespace_dict:
            nso = self.namespace_dict[ns]
            newcfg = nso.as_iface(self.namespace_dict, all_deps[ns])
            newfname = jp(newdir, ns.replace('::', '__') + '.i.tmpl.new')
            nso.write(newcfg, newfname)
            if ns in swig_interface.ignore_ns:
                continue
            oldfname = jp(olddir, ns.replace('::', '__') + '.i.tmpl')
            oldcfg = nso.from_iface(oldfname)
            if oldcfg:
                # compare both ways
                compare(ns, (newcfg, newfname), (oldcfg, oldfname), swig_interface.allowed_diffs, [])
                compare(ns, (oldcfg, oldfname), (newcfg, newfname), swig_interface.allowed_diffs, swig_interface.ignores)
            elif not nso.is_empty():
                print('Error: Could not find file ' + oldfname + ' or config in file.')


###############################################################################
###############################################################################
"""
FIXME
- template template and jinja macros need to work with all the data we extract
  - like generating imports and "#define USE_*" from deps
  - when producing the first versions for all the algorithms
    - think about simplifications, like converting all method values to lists
- almost everything we do manually in the template bodies should be moved
  to the cfg dicts, this allows much easier changes to all files.
  Moreover it will improve what we can do to compare C++ and our templates
- see which sections we currently do not compare and include in comparison
- some manual add_compute stuff can probably be simplified with the new tuple syntax
  for templates (see kmeans__init).
"""
###############################################################################
###############################################################################
# def write_R(self):
#     def renv(ns, pns):
#         if ns != 'daal':
#             pns = pns.replace('::', '$')
#             print('daal$'+ pns + '$' + ns.rsplit('::', 1)[-1] + ' <- new.env(parent = daal$' + pns + ')')
#         for c in self.namespace_dict[ns].children:
#             renv(c, ns)
#     print('daal <- new.env()')
#     renv('daal', 'environment()')
