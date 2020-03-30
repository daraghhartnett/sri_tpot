import logging
from uuid import uuid4
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from d3m.container import DataFrame
from .pipelinecache import PipelineCache

_logger = logging.getLogger(__name__)

TRANSFORMER_FAMILIES = { 'FEATURE_SELECTION', 'DATA_PREPROCESSING', 'DATA_TRANSFORMATION', 'FEATURE_EXTRACTION' }
PREDICTOR_FAMILIES = { 'CLASSIFICATION', 'REGRESSION', 'TIME_SERIES_FORECASTING' }
PREDICTED_TARGET = 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'


class D3MWrapper(object):

    PIPELINE_CACHE = None
    FAMILY_HYPERPARAMETER_SETTINGS = {}

    @staticmethod
    def enable_cache(enable):
        if enable:
            D3MWrapper.PIPELINE_CACHE = PipelineCache()
        else:
            D3MWrapper.PIPELINE_CACHE = None

    @staticmethod
    def set_family_hyperpameters(family, **hps):
        try:
            family_entry = D3MWrapper.FAMILY_HYPERPARAMETER_SETTINGS[family]
        except KeyError:
            family_entry = D3MWrapper.FAMILY_HYPERPARAMETER_SETTINGS[family] = {}
        for hpkey, hpval in hps.items():
            family_entry[hpkey] = hpval

    def get_family_hyperpameters(self):
        try:
            return D3MWrapper.FAMILY_HYPERPARAMETER_SETTINGS[self._family]
        except KeyError:
            return {}

    def get_internal_primitive(self):
        return self._prim

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
    def add_class(cls, oclass, opath):
        cname = oclass.__name__
        cls.wrapped_classes[cname] = oclass
        cls.class_paths[cname] = opath
        cls.path_classes[opath] = oclass

    @classmethod
    def get_class_from_name(cls, cname):
        return cls.wrapped_classes[cname]

    @classmethod
    def get_class_from_path(cls, cname):
        return cls.path_classes[cname]

    @classmethod
    def get_path(cls, cname):
        return cls.class_paths[cname]

    @classmethod
    def have_class(cls, cname):
        return cname in cls.wrapped_classes


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

        # Family settings
        family_mods = self.get_family_hyperpameters()
        for key, val in family_mods.items():
            if key in hpdefaults:
                if self.takes_hyperparameter_value(key, val):
                    hpmods[key] = val
            else:
                if not self._logged_hp_warnings:
                    _logger.info("Warning: {} does not accept the {} family hyperpameter".format(pclass, key))

        # Local settings
        for key, val in params.items():
            if isinstance(val, D3MWrapper):
                val = val.get_internal_primitive()
            if key in hpdefaults:
                if self.takes_hyperparameter_value(key, val):
                    hpmods[key] = val
            else:
                if not self._logged_hp_warnings:
                    _logger.info("Warning: {} does not accept the {} hyperparam".format(pclass, key))

        self.__class__._logged_hp_warnings = True
        return hpmods
    config['_get_hpmods'] = _get_hpmods

    def __init__(self, **kwargs):
        self._pclass = pclass
        self._params = kwargs
        self._fitted = False
        self._family = family
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
        if self._get_cached_produce(X) is not None:
            return self
        required_kwargs = mdata['primitive_code']['instance_methods']['set_training_data']['arguments']
        supplied_kwargs = {}
        if 'inputs' in required_kwargs:
            if not isinstance(X, DataFrame):
                X = DataFrame(X, generate_metadata=True)
            supplied_kwargs['inputs'] = X
        if 'outputs' in required_kwargs:
            if not isinstance(y, DataFrame):
                y = DataFrame(y, generate_metadata=True)
            supplied_kwargs['outputs'] = y
        self._prim.set_training_data(**supplied_kwargs)
        self._prim.fit()
        self._fitted = True
        return self
    config['fit'] = fit

    def transform(self, X):
        result = self._get_cached_produce(X)
        if result is not None:
            return result
        if not isinstance(X, DataFrame):
            X = DataFrame(X, generate_metadata=True)
        result = self._prim.produce(inputs=X).value
        return self._cache_produce(X, result)
    if family in TRANSFORMER_FAMILIES:
        config['transform'] = transform

    def predict(self, X):
        if not isinstance(X, DataFrame):
            X = DataFrame(X, generate_metadata=True)
        result = self._get_cached_produce(X)
        if result is not None:
            return result
        df = self._prim.produce(inputs=X).value
        pred_columns = df.metadata.get_columns_with_semantic_type(PREDICTED_TARGET)
        if len(pred_columns) == 0:  # Punt
            result = df.values
        else:
            result = df.iloc[:, pred_columns[0]].values
        self._cache_produce(X, result)
        return result
    if family in PREDICTOR_FAMILIES:
        config['predict'] = predict

    def get_internal_class(self):
        return pclass
    config['get_internal_class'] = classmethod(get_internal_class)

    def get_internal_primitive(self):
        return self._prim
    config['get_internal_primitive'] = get_internal_primitive

    def _get_cache_metadata(self, X, add=False):
        if D3MWrapper.PIPELINE_CACHE is None:
            return None
        df_mdata = X.metadata.query(())
        try:
            data_id = mdata['autoflow_dataset_id']
            pipeline_steps = mdata['autoflow_pipeline_steps']
        except KeyError:
            if add:
                _logger.info("Adding autoflow metadata for step %s" % str(self))
                data_id = str(uuid4())
                pipeline_steps = ()
                df_mdata = dict(**df_mdata, autoflow_dataset_id=data_id, autoflow_pipeline_steps=pipeline_steps)
                X.metadata = X.metadata.update((), df_mdata)
            else:
                return None
        return data_id, pipeline_steps
    config['_get_cache_metadata'] = _get_cache_metadata

    def _get_cached_produce(self, X):
        df_mdata = self._get_cache_metadata(X)
        if df_mdata is None:
            return None
        data_id, pipeline_steps = df_mdata
        pipeline = pipeline_steps + ( str(self), )
        return D3MWrapper.PIPELINE_CACHE.retrieve_from_cache(data_id, pipeline)
    config['_get_cached_produce'] = _get_cached_produce

    def _cache_produce(self, input, output):
        if D3MWrapper.PIPELINE_CACHE is None:
            return output
        input_mdata = input.metadata.query(())
        # The input lacks the appropriate metadata.  Add it.
        try:
            data_id = input_mdata['autoflow_dataset_id']
            pipeline_steps = input_mdata['autoflow_pipeline_steps']
        except KeyError:
            _logger.info("Adding autoflow metadata for step %s" % str(self))
            data_id = str(uuid4())
            pipeline_steps = ()
            input_mdata = dict(**input_mdata, autoflow_dataset_id=data_id, autoflow_pipeline_steps=pipeline_steps)
            input.metadata = input.metadata.update((), input_mdata)
        # Now add the new step and update the output metadata
        current_step = str(self)
        pipeline = pipeline_steps + ( current_step, )
        output_mdata = output.metadata.query(())
        output.metadata = output.metadata.update((), dict(**output_mdata,
                                                          autoflow_dataset_id=data_id,
                                                          autoflow_pipeline_steps=pipeline))
        # Finally store the output in the cache
        D3MWrapper.PIPELINE_CACHE.add_to_cache(data_id, pipeline, output)
        return output
    config['_cache_produce'] = _cache_produce

    # Special method to enable TPOT to suppress unsupported arg primitives
    @staticmethod
    def takes_hyperparameter(hp):
        return hp in hpdefaults
    config['takes_hyperparameter'] = takes_hyperparameter

    # Check whether the primitive accepts a value that TPOT thinks is valid
    def takes_hyperparameter_value(self, hp, value):
        try:
            hpclass.configuration[hp].validate(value)
            return True
        except:
            if not self._logged_hp_warnings:
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
    class_._logged_hp_warnings = False

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
