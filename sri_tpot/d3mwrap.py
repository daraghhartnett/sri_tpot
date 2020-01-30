import logging
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from d3m.container import DataFrame
from .pipelinecache import PipelineCache

_logger = logging.getLogger(__name__)


TRANSFORMER_FAMILIES = {'FEATURE_SELECTION', 'DATA_PREPROCESSING', 'DATA_TRANSFORMATION', 'FEATURE_EXTRACTION'}
PREDICTOR_FAMILIES = { 'CLASSIFICATION', 'REGRESSION', 'TIME_SERIES_FORECASTING' }
PREDICTED_TARGET = 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'

class D3MWrapper(object):

    PIPELINE_CACHE = None

    @staticmethod
    def enable_cache(enable):
        if enable:
            D3MWrapper.PIPELINE_CACHE = PipelineCache()
        else:
            D3MWrapper.PIPELINE_CACHE = None

    def __str__(self):
        cname = self.__class__.__name__
        hpmods = self._hpmods
        hps = sorted(hpmods.keys())
        return "%s(%s)" % (cname, ", ".join(["%s=%s" % (k, hpmods[k]) for k in hps]))

    def dataset_key(self, data):
        return str(data.id())


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
        self._hpmods = self._get_hpmods(kwargs)
        self._prim = pclass(hyperparams=hpclass(hpdefaults, **self._hpmods))
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
    if family in TRANSFORMER_FAMILIES:
        config['transform'] = transform

    def predict(self, X):
        # We convert to ndarray here, because sklearn gets confused about Dataframes
#        print("%s asked to predict on data with %d rows" % (type(self), len(X)))
        df = self._prim.produce(inputs=DataFrame(X, generate_metadata=False)).value
        # Find the column with predicted values
        pred_columns = df.metadata.get_columns_with_semantic_type(PREDICTED_TARGET)
        if len(pred_columns) == 0:  # Punt
            result = df.values
        else:
            result = df.iloc[:,pred_columns[0]].values
#        print("%s produced %d predictions" % (type(self), len(result)))
        return result
    if family in PREDICTOR_FAMILIES:
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
    if family == 'REGRESSION' or family == 'TIME_SERIES_FORECASTING':
        parents.append(RegressorMixin)
    if family == 'CLASSIFICATION':
        parents.append(ClassifierMixin)
    if family in TRANSFORMER_FAMILIES:
        parents.append(TransformerMixin)
    class_ = type(newname, tuple(parents), config)
    class_.pclass = pclass

    D3MWrappedOperators.add_class(class_, ppath)
    # For pickling to work, we need to install the class globally
    globals()[newname] = class_

    return class_


def supports_hyperparameter(obj, pname):
    """
    Safety check to see whether an argument specified in the config
    file is actually supported.  Depending on how an sklearn class
    was wrapped, some of its hyperparameters may not be exposed.
    """
    if issubclass(obj, D3MWrapper):
        return obj.takes_hyperparameter(pname)
    else:
        return True


def supports_hyperparameter_setting(obj, pname, value):
    """
    Safety check to make sure an argument value that TPOT considers
    valid is actually supported by a D3M primitive.
    """
    if issubclass(obj, D3MWrapper):
        return obj.takes_hyperparameter_value(pname, value)
    else:
        return True
