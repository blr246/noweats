"""
Analysis of raw counts extracted from tweets.
"""
from collections import Counter, defaultdict
from nltk.metrics import edit_distance
import itertools as it
import math


def merge_most_common_counts(counts, num_to_get=None,
                             simiarity_thresh=0.7, min_len=3, debug=False):
    """
    Consolidate counts for sufficiently similar things.

    N.B. This runs slowly on entire datasets.
    """

    if num_to_get is None:
        num_to_get = len(counts)

    if not isinstance(num_to_get, int) and num_to_get < 0:
        raise ValueError("Parameter num_to_get must be a positive number")

    merged = set()
    merged_counts = Counter()

    similarity_score = \
        lambda k1, k2: 1. - (
            edit_distance(k1, k2) / float(max(len(k1), len(k2))))

    # Filter keys by minimum length. For each key, compute similarity ratio to
    # all other keys and merge sets based on threshold. Use key from the
    # largest subgroup of the merge.

    filtered_keys = [(key, key.lower())
                     for key, _ in counts.most_common()
                     if len(key) >= min_len]

    for i, (ikey, ikeylower) in enumerate(filtered_keys):

        # Skip when merged already.
        if ikey in merged:
            continue

        merged.add(ikey)

        # Stop when we have the desired number of keys.
        if len(merged_counts) == num_to_get:
            break

        # Find similar jkeylower.

        keys_to_merge = [jkey
                         for jkey, jkeylower in filtered_keys[i + 1:]
                         if jkey not in merged
                         and similarity_score(ikeylower,
                                              jkeylower) > simiarity_thresh]

        key, count = ikey, counts[ikey]
        max_count = count

        # Remove these keys as we merge them, keeping the key name as the
        # largest count.

        for jkey in keys_to_merge:

            merged.add(jkey)

            key_count = counts[jkey]
            count += key_count
            if key_count > max_count:
                max_count = key_count
                key = jkey

        if debug is True:
            print 'Merging {} <= {}'.format(key,
                                            tuple(it.chain([ikey],
                                                           keys_to_merge)))
        merged_counts[key] = count

    return merged_counts


def find_interesting(counts, num_to_find, max_words=None):
    """ Use TF/IDF scores to extract interesting foods. """

    if max_words is not None:
        len_filter = lambda (terms, f): len(terms) <= max_words
    else:
        len_filter = lambda x: x

    term_frequencies = defaultdict(int)

    food_terms = []
    foods = it.ifilter(
        len_filter, it.imap(lambda (k, f): (k.split(), f), counts.iteritems()))
    for terms, count in foods:
        food_terms.append(terms)
        for term in set(terms):
            term_frequencies[term] += count

    num_docs = float(len(counts))
    idf = dict((term, math.log(num_docs / freq))
               for term, freq in term_frequencies.iteritems())

    most_interesting = sorted(
        food_terms,
        key=lambda terms: sum(idf[t] for t in set(terms)),
        reverse=True)

    if num_to_find is None:
        return [' '.join(terms) for terms in most_interesting]
    else:
        return [' '.join(terms) for terms in most_interesting[:num_to_find]]
