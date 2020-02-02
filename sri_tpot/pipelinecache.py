from collections import namedtuple

CacheEntry = namedtuple("CacheEntry", ("key", "value", "reftime"))

class PipelineCache(object):

    """
    A class for caching intermediate processing results.
    """

    def __init__(self, max_entries=500, purge_method="lru"):
        self.max_entries = max_entries
        self.purge_method = purge_method
        self.cache = {}
        self.time = 0
        self.purge = getattr(self, purge_method)

    def retrieve_from_cache(self, dataset_id, pipeline):
        """
        Retrieves data indexed by the pipeline argument, if present.

        :param: dataset_id: A string representing the dataset against which the pipeline is executed
        :param pipeline: A sequence of strings, each representing a fully parameterized primitive
        :return: The result from a previous execution of this pipeline, if present, else None.
        """
        self.time += 1
        key = (dataset_id, *pipeline)
        try:
            entry = self.cache[key]
            entry.reftime = self.time
            return entry.value
        except KeyError:
            return None

    def add_to_cache(self, dataset_id, pipeline, data):
        """
        Stores the provided data at the position indexed by pipeline.

        :param: dataset_id: A string representing the dataset against which the pipeline is executed
        :param pipeline:  A sequence of strings, each representing a fully parameterized primitive
        :param data: The result of executing the corresponding pipeline.
        :return: None
        """
        self.time += 1
        key = (dataset_id, *pipeline)
        self.purge(data)
        self.cache[key] = CacheEntry(key=key, value=data, reftime=self.time)

    def cache_is_full(self, data=None):
        """
        Determines whether the cache is full or will be made full by storing the indicated data.

        :param data: A value we wish to store in the cache
        :return: True if the cache is full, or if storing the provided result (if not None) would make it full.
        """
        count = len(self.cache)
        if data is not None:
            count += 1
        return count > self.max_entries

    def lru(self, data=None):
        """
        Optionally eliminates information stored in the cache if it's at capacity (or will be above capacity after
        storing the provided data).

        :param data: A value we wish to store in the cache
        :return: None
        """
        if not self.cache_is_full(data):
            return
        entries = sorted(self.cache.values(), key=lambda x: x.reftime)
        while self.cache_is_full(data):
            entry = entries.pop(0)
            del self.cache[entry.key]

    def clear(self):
        """
        Empties the cache.

        :return: None
        """
        self.cache = {}
        self.time = 0



