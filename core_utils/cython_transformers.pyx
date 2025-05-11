# cython: language_level=3

cpdef list transform_search_results(object raw_results):
    """
    Convert raw search result tuples into a list of dicts.
    raw_results: iterable of (id, score, metadata)
    """
    cdef list out = []
    for r in raw_results:
        out.append({'id': r[0], 'score': r[1], 'metadata': r[2]})
    return out 