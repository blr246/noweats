"""
Model of English language.
"""
from noweats.extraction import allowed_chars_no_whitespace, \
    sentence_split_clean_data
from nltk import word_tokenize
from collections import Counter

import math
import itertools as its


def word_to_bag(word):
    """ Convert word to bag-of-chars. """
    return ''.join(sorted(set(word)))


def make_en_prefix_suffix_model(model):
    """ Create a function that scores English words. """

    log_tform = lambda m: dict((k, math.log(v)) for k, v in m.iteritems())

    prefixes, suffixes, bags = model
    prefixes = log_tform(prefixes)
    suffixes = log_tform(suffixes)
    bags = log_tform(bags)

    alphabet = allowed_chars_no_whitespace()

    actual_alphabet = set(char for key in
                          its.chain(prefixes.iterkeys(),
                                    suffixes.iterkeys(),
                                    bags.iterkeys())
                          for char in key)
    if len(actual_alphabet) > len(alphabet):
        print "unexpected chars in alphabet {}".format(
            actual_alphabet - alphabet)
        alphabet = alphabet.union(actual_alphabet)

    num_chars = len(alphabet)

    tuple_len = max(len(k) for k in its.chain(prefixes.iterkeys(),
                                              suffixes.iterkeys()))

    # Tuples can lead or end with null chars but must have at least 1 non-null.
    possible_tuples = (num_chars + 1)**(tuple_len - 1) * num_chars
    # Exclude the empty bag.
    possible_bags = 2**num_chars - 1

    norm_prefix = -math.log(sum(prefixes.itervalues()) + possible_tuples)
    norm_suffix = -math.log(sum(suffixes.itervalues()) + possible_tuples)
    norm_bag = -math.log(sum(bags.itervalues()) + possible_bags)

    def p_word(word):
        """ Compute probability of a word under the model. """

        word_lower = word.lower()
        prefix, suffix = word_lower[:tuple_len], word_lower[-tuple_len:]
        bag = word_to_bag(word)

        ll_prefix = prefixes[prefix] if prefix in prefixes else 1
        ll_suffix = suffixes[suffix] if suffix in suffixes else 1
        ll_bag = bags[bag] if bag in bags else 1

        return ll_prefix + norm_prefix \
            + ll_suffix + norm_suffix \
            + ll_bag + norm_bag

    return p_word


def expectation_en_tweet(tweet, p_word, is_lower=False):
    """
    Compute expected probability that a tweet is English.
    """

    return math.log(sum(math.exp(expectation_en_sentence(s, p_word, is_lower))
                        for s in tweet) / len(tweet))


def expectation_en_sentence(sentence, p_word, is_lower=False):
    """
    Compute expected probability that a word from the sentence is English.
    """

    if is_lower is True:
        scorer = p_word
    else:
        scorer = lambda w: p_word(w.lower())

    if isinstance(sentence, str):
        sentence = sentence.split()

    return math.log(sum(math.exp(scorer(w)) for w in sentence) / len(sentence))


def likelihood_en_tweet(tweet, p_word, is_lower=False):
    """
    Compute likelihood that a tweet is English.
    """

    return sum(likelihood_en_sentence(s, p_word, is_lower)
               for s in tweet) / len(tweet)


def likelihood_en_sentence(sentence, p_word, is_lower=False):
    """
    Compute likelihood that a word from the sentence is English.
    """

    if is_lower is True:
        scorer = p_word
    else:
        scorer = lambda w: p_word(w.lower())

    if isinstance(sentence, str):
        sentence = sentence.split()

    return sum(scorer(w) for w in sentence)


def build_en_prefix_suffix_model(data_json):
    """
    Create a probabilisitc model of English words from data.

    Features:
        3-prefixes, 3-suffixes, bag-of-chars model.

    For the 3-prefixes and 3-suffixes, there are |chars|(|chars| + 1)^2
    possible observations. For the bag-of-chars, there are 2^|chars| possible
    bags.

    During inference, we want to avoid the partition function (i.e. all
    sentences with 3 words). Therefore, we compute the expectation that a word
    in the sentence is English and threshold that.
    """

    all_tweets = sentence_split_clean_data(data_json, [''])
    toks = [t.lower() for tw in all_tweets
            for s in tw
            for t in word_tokenize(s)]
    prefixes = Counter(t[:3] for t in toks)
    suffixes = Counter(t[-3:] for t in toks)
    bags = Counter(word_to_bag(t) for t in toks)

    return [prefixes, suffixes, bags]
