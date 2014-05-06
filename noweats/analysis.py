"""
Analysis of raw counts extracted from tweets.
"""
from collections import Counter, defaultdict
from nltk.metrics import edit_distance
import itertools as it
import math


def merge_most_common_counts(counts, num_to_get=None,
                             simiarity_thresh=0.7,
                             len_range=(3, 30), debug=False):
    """
    Consolidate counts for sufficiently similar things.

    N.B. This runs slowly on entire datasets.
    """

    if num_to_get is None:
        num_to_get = len(counts)

    if not isinstance(num_to_get, int) and num_to_get < 0:
        raise ValueError("Parameter num_to_get must be a positive number")

    similarity_score = \
        lambda k1, k2: 1. - (
            edit_distance(k1, k2) / float(max(len(k1), len(k2))))

    # Filter keys by minimum length. For each key, compute similarity ratio to
    # all other keys and merge sets based on threshold. Use key from the
    # largest subgroup of the merge.

    filtered_keys = [key.lower()
                     for key, _ in counts.most_common()
                     if len(key) >= len_range[0] and len(key) <= len_range[1]]

    merged = [None] * len(filtered_keys)
    merged_counts = Counter()

    for i, ikey in enumerate(filtered_keys):

        # Skip when merged already.
        if merged[i] is not None:
            continue

        merged[i] = i

        # Stop when we have the desired number of keys.
        if len(merged_counts) == num_to_get:
            break

        # Find similar jkeylower.

        keys_to_merge = [
            (j, jkey)
            for j, jkey in enumerate(filtered_keys[i + 1:], start=i + 1)
            if merged[j] is None
            and similarity_score(ikey, jkey) > simiarity_thresh]

        key, count = ikey, counts[ikey]
        max_count = count

        # Remove these keys as we merge them, keeping the key name as the
        # largest count.

        for j, jkey in keys_to_merge:

            merged[j] = j

            key_count = counts[jkey]
            count += key_count
            if key_count > max_count:
                max_count = key_count
                key = jkey

        if debug is True:
            print 'Merging {} <= {}'.format(
                key, tuple(it.chain([ikey],
                                    it.imap(lambda (_, k): k,
                                            keys_to_merge))))
        merged_counts[key] = count

    return merged_counts


def find_interesting(counts, num_to_find,
                     max_words=None, match_thresh=0.6, debug=False):
    """ Use TF/IDF scores to extract interesting foods. """

    def match_score(foodi, foodj):
        """ Score simiarity between two foods. """
        words_matched = sum(1 for _ in it.takewhile(lambda (a, b): a == b,
                                                    it.izip(foodi, foodj)))

        if words_matched > 0:
            i_len_add = sum(len(w) for w in foodi[words_matched:])
            j_len_add = sum(len(w) for w in foodj[words_matched:])

            match_len = words_matched - 1 + sum(
                len(w) for w in foodi[:words_matched])
            i_len = match_len + i_len_add + len(foodi) - words_matched
            j_len = match_len + j_len_add + len(foodj) - words_matched
            score = float(match_len) / max(i_len, j_len)
            return score

        else:
            return 0

    if max_words is not None:
        len_filter = lambda (terms, f): len(terms) <= max_words
    else:
        len_filter = lambda x: x

    term_frequencies = defaultdict(int)

    food_terms = []
    foods = it.ifilter(
        len_filter, it.imap(lambda (k, f): (tuple(k.split()), f),
                            counts.iteritems()))
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

    merged_most_interesting = []
    merged = [None] * len(most_interesting)
    for i, foodi in enumerate(most_interesting):

        if merged[i] is not None:
            continue
        merged[i] = i

        food = ' '.join(foodi)

        # Find matches.
        foods_to_merge = [
            (j, foodj)
            for j, foodj in enumerate(most_interesting[i + 1:], start=i + 1)
            if merged[j] is None
            and match_score(foodi, foodj) > match_thresh
        ]

        # Mark foods merged.
        for j, foodj in foods_to_merge:
            merged[j] = j

        if debug is True:
            print 'Merging {} <= {}'.format(
                food, tuple(it.chain([' '.join(foodi)],
                                     it.imap(lambda (_, f): ' '.join(f),
                                             foods_to_merge))))

        # This foodi had the highest score, so keep it.
        merged_most_interesting.append(food)

    if num_to_find is None:
        return merged_most_interesting
    else:
        return merged_most_interesting[:num_to_find]
