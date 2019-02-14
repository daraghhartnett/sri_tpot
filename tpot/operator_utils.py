# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import d3m.index
from d3m.container import DataFrame
import tpot.d3mgrid as d3mgrid
import inspect

import logging

_logger = logging.getLogger(__name__)

class D3MWrapper(object):
    pass


class D3MWrappedOperators(object):

    wrapped_classes = {}
    class_paths = {}
    path_classes = {}
    
    @classmethod
    def add_class(self, oclass, opath):
        cname = oclass.__name__
        self.wrapped_classes[cname] = oclass
        self.class_paths[cname] = opath
        self.path_classes[opath] = oclass

    @classmethod
    def get_class_from_name(self, cname):
        return self.wrapped_classes[cname]

    @classmethod
    def get_class_from_path(self, cname):
        return self.path_classes[cname]

    @classmethod
    def get_path(self, cname):
        return self.class_paths[cname]

    @classmethod
    def have_class(self, cname):
        return cname in self.wrapped_classes


class Operator(object):
    """Base class for operators in TPOT."""

    root = False  # Whether this operator type can be the root of the tree
    import_hash = None
    sklearn_class = None
    arg_types = None


class ARGType(object):
    """Base class for parameter specifications."""

    pass


def source_decode(sourcecode):
    """Decode operator source and import operator class.

    Parameters
    ----------
    sourcecode: string
        a string of operator source (e.g 'sklearn.feature_selection.RFE')


    Returns
    -------
    import_str: string
        a string of operator class source (e.g. 'sklearn.feature_selection')
    op_str: string
        a string of operator class (e.g. 'RFE')
    op_obj: object
        operator class (e.g. RFE)

    """
    tmp_path = sourcecode.split('.')
    op_str = tmp_path.pop()
    import_str = '.'.join(tmp_path)
    try:
        if sourcecode.startswith('tpot.'):
            exec('from {} import {}'.format(import_str[4:], op_str))
            op_obj = eval(op_str)
        elif sourcecode.startswith('d3m.primitives'):
            exec('from {} import {}'.format(import_str, op_str))
            op_obj = D3MWrapperClassFactory(eval(op_str), sourcecode)
            op_str = op_obj.__name__
        else:
            exec('from {} import {}'.format(import_str, op_str))
            op_obj = eval(op_str)
            
            
    except ImportError:
        _logger.info('Warning: {} is not available and will not be used by TPOT.'.format(sourcecode))
        op_obj = None

    return import_str, op_str, op_obj


def set_sample_weight(pipeline_steps, sample_weight=None):
    """Recursively iterates through all objects in the pipeline and sets sample weight.

    Parameters
    ----------
    pipeline_steps: array-like
        List of (str, obj) tuples from a scikit-learn pipeline or related object
    sample_weight: array-like
        List of sample weight
    Returns
    -------
    sample_weight_dict:
        A dictionary of sample_weight

    """
    sample_weight_dict = {}
    if not isinstance(sample_weight, type(None)):
        for (pname, obj) in pipeline_steps:
            if inspect.getargspec(obj.fit).args.count('sample_weight'):
                step_sw = pname + '__sample_weight'
                sample_weight_dict[step_sw] = sample_weight

    if sample_weight_dict:
        return sample_weight_dict
    else:
        return None


def ARGTypeClassFactory(classname, prange, BaseClass=ARGType):
    """Dynamically create parameter type class.

    Parameters
    ----------
    classname: string
        parameter name in a operator
    prange: list
        list of values for the parameter in a operator
    BaseClass: Class
        inherited BaseClass for parameter

    Returns
    -------
    Class
        parameter class

    """
    return type(classname, (BaseClass,), {'values': prange})


def TPOTOperatorClassFactory(opsourse, opdict, BaseClass=Operator, ArgBaseClass=ARGType, operator_advice="strict"):
    """Dynamically create operator class.

    Parameters
    ----------
    opsourse: string
        operator source in config dictionary (key)
    opdict: dictionary
        operator params in config dictionary (value)
    regression: bool
        True if it can be used in TPOTRegressor
    classification: bool
        True if it can be used in TPOTClassifier
    BaseClass: Class
        inherited BaseClass for operator
    ArgBaseClass: Class
        inherited BaseClass for parameter
    operator_advice: string
        controls how terminals are generated (see base.py for more info)

    Returns
    -------
    op_class: Class
        a new class for a operator
    arg_types: list
        a list of parameter class

    """
    class_profile = {}
    dep_op_list = {} # list of nested estimator/callable function
    dep_op_type = {} # type of nested estimator/callable function
    import_str, op_str, op_obj = source_decode(opsourse)

    if not op_obj:
        return None, None

    if _can_be_root(op_obj):
        class_profile['root'] = True
        optype = "Classifier or Regressor"
    else:
        optype = "Preprocessor or Selector"

    @classmethod
    def op_type(cls):
        """Return the operator type.
        
        Possible values:
        "Classifier", "Regressor", "Selector", "Preprocessor"
        """
        return optype

    class_profile['type'] = op_type
    class_profile['sklearn_class'] = op_obj
    import_hash = {}
    import_hash[import_str] = [op_str]

    def _make_arg_types(pname, prange):
        arg_types = []
        if not isinstance(prange, dict):
            prange = [ v for v in prange if _supports_arg_setting(op_obj, pname, v) ]
            if len(prange) == 0:
                _logger.info("Warning: No valid values provided for {} of {}".format(pname, op_obj.__name__))
                return None
            classname = '{}__{}'.format(op_str, pname)
            arg_types.append(ARGTypeClassFactory(classname, prange, ArgBaseClass))
        else:
            for dkey, dval in prange.items():
                dep_import_str, dep_op_str, dep_op_obj = source_decode(dkey)
                if dep_import_str in import_hash:
                    import_hash[import_str].append(dep_op_str)
                else:
                    import_hash[dep_import_str] = [dep_op_str]
                dep_op_list[pname] = dep_op_str
                dep_op_type[pname] = dep_op_obj
                if dval:
                    for dpname in sorted(dval.keys()):
                        dprange = dval[dpname]
                        classname = '{}__{}__{}'.format(op_str, dep_op_str, dpname)
                        arg_types.append(ARGTypeClassFactory(classname, dprange, ArgBaseClass))
        return arg_types

    def _discover_arg_values(hpname):
        pclass = op_obj.get_internal_class()
        mdata = pclass.metadata.query()
        hpsclass = mdata['primitive_code']['class_type_arguments']['Hyperparams']
        hpclass = hpsclass.configuration[hpname]
        print("discover_arg_values: %s of %s" % (hpname, pclass))
        return d3mgrid.grid(hpclass)

    def _discover_arg_types():
        arg_types = []
        pclass = op_obj.get_internal_class()
        mdata = pclass.metadata.query()
        hpsclass = mdata['primitive_code']['class_type_arguments']['Hyperparams']
        for pname, hp in hpclass.configuration.items():
            prange = _discover_arg_values(pname)
            if prange is None:
                continue
            parg_types = _make_arg_types(pname, prange)
            if parg_types is None:
                continue
            arg_types.extend(parg_types)
        return arg_types

    def _read_arg_types(strict):
        arg_types = []
        for pname in sorted(opdict.keys()):
            if not _supports_arg(op_obj, pname):
                continue
            if strict:
                prange = opdict[pname]
            else:
                prange = _discover_arg_values(pname)
                print("discover_arg_values: %s" % prange)
            if prange is None:
                continue
            parg_types = _make_arg_types(pname, prange)
            if parg_types is None:
                continue
            arg_types.extend(parg_types)
        return arg_types

    if operator_advice == "discover":
        arg_types = _discover_arg_types()
    else:
        arg_types = _read_arg_types(operator_advice=="strict")

    class_profile['arg_types'] = tuple(arg_types)
    class_profile['import_hash'] = import_hash
    class_profile['dep_op_list'] = dep_op_list
    class_profile['dep_op_type'] = dep_op_type

    @classmethod
    def parameter_types(cls):
        """Return the argument and return types of an operator.

        Parameters
        ----------
        None

        Returns
        -------
        parameter_types: tuple
        Tuple of the DEAP parameter types and the DEAP return type for the
        operator

        """
        return ([np.ndarray] + arg_types, np.ndarray)

    class_profile['parameter_types'] = parameter_types

    @classmethod
    def export(cls, *args):
        """Represent the operator as a string so that it can be exported to a file.

        Parameters
        ----------
        args
        Arbitrary arguments to be passed to the operator

        Returns
        -------
        export_string: str
        String representation of the sklearn class with its parameters in
        the format:
        SklearnClassName(param1="val1", param2=val2)
        
        """
        op_arguments = []

        if dep_op_list:
            dep_op_arguments = {}

        for arg_class, arg_value in zip(arg_types, args):
            aname_split = arg_class.__name__.split('__')
            if isinstance(arg_value, str):
                arg_value = '\"{}\"'.format(arg_value)
            if len(aname_split) == 2:  # simple parameter
                op_arguments.append("{}={}".format(aname_split[-1], arg_value))
            # Parameter of internal operator as a parameter in the
            # operator, usually in Selector
            else:
                if aname_split[1] not in dep_op_arguments:
                    dep_op_arguments[aname_split[1]] = []
                dep_op_arguments[aname_split[1]].append("{}={}".format(aname_split[-1], arg_value))

        tmp_op_args = []
        if dep_op_list:
            # To make sure the inital operators is the first parameter just
            # for better persentation
            for dep_op_pname, dep_op_str in dep_op_list.items():
                arg_value = dep_op_str # a callable function, e.g scoring function
                doptype = dep_op_type[dep_op_pname]
                if _is_estimator(doptype):
                        arg_value = "{}({})".format(dep_op_str, ", ".join(dep_op_arguments[dep_op_str]))
                tmp_op_args.append("{}={}".format(dep_op_pname, arg_value))
        op_arguments = tmp_op_args + op_arguments
        return "{}({})".format(op_obj.__name__, ", ".join(op_arguments))

    class_profile['export'] = export

    @classmethod
    def instance(cls, *args):
        """
        Create an instance of the wrapped class, parameterized with the indicated args.
        """
        op_arguments = []

        if dep_op_list:
            dep_op_arguments = {}

        for arg_class, arg_value in zip(arg_types, args):
            aname_split = arg_class.__name__.split('__')
            param = aname_split[1]
            if len(aname_split) == 2:  # simple parameter
                op_arguments.append((param, arg_value))
            # Parameter of internal operator as a parameter in the
            # operator, usually in Selector
            else:
                if param not in dep_op_arguments:
                    dep_op_arguments[param] = []
                dep_op_arguments[param].append((aname_split[-1], arg_value))

        tmp_op_args = []
        if dep_op_list:
            # To make sure the inital operators is the first parameter just
            # for better persentation
            for dep_op_pname, dep_op_str in dep_op_list.items():
                arg_value = dep_op_str # a callable function, e.g scoring function
                doptype = dep_op_type[dep_op_pname]
                if _is_estimator(doptype):
                    depargs = dict(dep_op_arguments[dep_op_str])
                    arg_value = dep_op_str(**depargs)
                tmp_op_args.append((dep_op_pname, arg_value))
        op_arguments = dict(tmp_op_args + op_arguments)
        return op_obj(**op_arguments)

    class_profile['instance'] = instance

    op_classname = 'TPOT_{}'.format(op_str)
    op_class = type(op_classname, (BaseClass,), class_profile)
    op_class.__name__ = op_str
    return op_class, arg_types


def D3MWrapperClassFactory(pclass, ppath):
    """
    Generates a wrapper class for D3M primitives to make them behave
    like standard sklearn estimators.

    Parameters
    ----------
    pclass: Class
       The class object for a D3M primitive.

    Returns
    -------
    A newly minted class that is compliant with the sklearn estimator
    API and delegates to the underlying D3M primitive.
    """

    mdata = pclass.metadata.query()
    hpclass = mdata['primitive_code']['class_type_arguments']['Hyperparams']
    hpdefaults = hpclass.defaults()
    family = mdata['primitive_family']

    config = {}

    def _get_hpmods(self, params):
        hpmods = {}
        for key, val in params.items():
            if isinstance(val, D3MWrapper):
                val = val.get_internal_primitive()
            if key in hpdefaults:
                hpmods[key] = val
            else:
                _logger.info("Warning: {} does not accept the {} hyperparam".format(pclass, key))
        # The default true setting wreaks havoc on our ability to do cross-validation
#        hpmods['add_index_columns'] = False
        return hpmods
    config['_get_hpmods'] = _get_hpmods

    def __init__(self, **kwargs):
        self._pclass = pclass
        self._params = kwargs
        self._fitted = False
        hpmods = self._get_hpmods(kwargs)
        self._prim = pclass(hyperparams=hpclass(hpdefaults, **hpmods))
    config['__init__'] = __init__

    def __get_state__(self):      
        if not self._fitted:
            self._prim = None
        return self.__dict__.copy()
    config['__getstate__'] = __get_state__

    def __set_state__(self, state):
        self.__dict__.update(state)
        if self._prim is None:
            hpmods = self._get_hpmods(self._params)
            self._prim = self._pclass(hyperparams=hpclass(hpdefaults, **hpmods))
    config['__setstate__'] = __set_state__

#    def __get_state__(self):
#        return self.__dict__.copy()
#    config['__get_state__'] = __get_state__
#
#    def __set_state__(self, state):
#        self.__dict__.update(state)
#    config['__set_state__'] = __set_state__

    # This is confusing: what sklearn calls params, d3m calls hyperparams
    def get_params(self, deep=False):
        return self._params
    config['get_params'] = get_params

    # Note that this blows away the previous underlying primitive.
    # Should be OK, since we only call this method before fitting.
    def set_params(self, **params):
        self._prim = pclass(hyperparams=hpclass(hpdefaults, **params))
    config['set_params'] = set_params

    def fit(self, X, y):
        required_kwargs = mdata['primitive_code']['instance_methods']['set_training_data']['arguments']
        supplied_kwargs = {}
        if 'inputs' in required_kwargs:
            supplied_kwargs['inputs'] = DataFrame(X, generate_metadata=False)
        if 'outputs' in required_kwargs:
            supplied_kwargs['outputs'] = DataFrame(y, generate_metadata=False)
        self._prim.set_training_data(**supplied_kwargs)
        self._prim.fit()
        self._fitted = True
        return self
    config['fit'] = fit

    def transform(self, X):
#        print("%s asked to transform data with %d rows" % (type(self), len(X)))
        result = self._prim.produce(inputs=DataFrame(X, generate_metadata=False)).value
#        print("%s transformed to %d rows" % (type(self), len(result)))
        return result
    if family == 'FEATURE_SELECTION' or family == 'DATA_PREPROCESSING' or family == 'DATA_TRANSFORMATION':
        config['transform'] = transform

    def predict(self, X):
        # We convert to ndarray here, because sklearn gets confused about Dataframes
#        print("%s asked to predict on data with %d rows" % (type(self), len(X)))
        result = self._prim.produce(inputs=DataFrame(X, generate_metadata=False)).value.values
#        print("%s produced %d predictions" % (type(self), len(result)))
        return result
    if family == 'CLASSIFICATION' or family == 'REGRESSION':
        config['predict'] = predict

    def get_internal_class(self):
        return pclass
    config['get_internal_class'] = classmethod(get_internal_class)

    def get_internal_primitive(self):
        return self._prim
    config['get_internal_primitive'] = get_internal_primitive

    # Special method to enable TPOT to suppress unsupported arg primitives
    @staticmethod
    def takes_hyperparameter(hp):
        return hp in hpdefaults
    config['takes_hyperparameter'] = takes_hyperparameter

    # Check whether the primitive accepts a value that TPOT thinks is valid
    @staticmethod
    def takes_hyperparameter_value(hp, value):
        try:
            hpclass.configuration[hp].validate(value)
            return True
        except:
            _logger.info("Warning: Suppressing value of {} for {} of {}".format(value, hp, pclass.__name__))
            return False
    config['takes_hyperparameter_value'] = takes_hyperparameter_value

    newname = 'AF_%s' % pclass.__name__
    parents = [D3MWrapper]
    if family == 'REGRESSION':
        parents.append(RegressorMixin)
    if family == 'CLASSIFICATION':
        parents.append(ClassifierMixin)
    if family == 'FEATURE_SELECTION' or family == 'DATA_PREPROCESSING' or family == 'DATA_TRANSFORMATION':
        parents.append(TransformerMixin)
    class_ = type(newname, tuple(parents), config)
    class_.pclass = pclass

    D3MWrappedOperators.add_class(class_, ppath)
    # For pickling to work, we need to install the class globally
    globals()[newname] = class_

    return class_

#####################################################################
# The functions below were added as part of the D3M compliance overhaul.
#####################################################################

def _can_be_root(optype):
    """
    'Root' in TPOT parlance means an operator class can serve as the 
    final stage of a pipeline.
    """
#    if issubclass(obj, D3MWrapper):
#        class_ = obj.pclass
#    else:
#        class_ = obj
    _logger.info("checking optype %s is subclass of ClassifierMixin or RegressorMixin" % optype)
    return (issubclass(optype, ClassifierMixin)
            or issubclass(optype, RegressorMixin))
#            or issubclass(class_, SupervisedLearnerPrimitiveBase))


def _is_estimator(optype):
    if inspect.isclass(optype): 
#        if issubclass(optype, D3MWrapper):
#            class_ = optype.pclass
#        else:
#            class_ = optype
        return (issubclass(optype, BaseEstimator) 
                or issubclass(optype, ClassifierMixin) 
                or issubclass(optype, RegressorMixin) 
                or issubclass(optype, TransformerMixin))
#                or issubclass(class_, SupervisedLearnerPrimitiveBase)
#                or issubclass(class_, TransformerPrimitiveBase))


def _supports_arg(obj, pname):
    """
    Safety check to see whether an argument specified in the config
    file is actually supported.  Depending on how an sklearn class
    was wrapped, some of its hyperparameters may not be exposed.
    """
    if issubclass(obj, D3MWrapper):
        return obj.takes_hyperparameter(pname)
    else:
        return True


def _supports_arg_setting(obj, pname, value):
    """
    Safety check to make sure an argument value that TPOT considers
    valid is actually supported by a D3M primitive.
    """
    if issubclass(obj, D3MWrapper):
        return obj.takes_hyperparameter_value(pname, value)
    else:
        return True
