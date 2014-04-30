"""
Extract foods that people are eating from their Tweets.
"""
from nltk.chunk import RegexpParser
from nltk import word_tokenize, pos_tag, Tree
from unidecode import unidecode
from HTMLParser import HTMLParser
from collections import Counter

import re
import json
import itertools as it
import bz2

_REMOVE_LINKS = '\\s?\\bhttp[\S]+'

_REMOVE_USER_HASH = '@[\\S]+\\s?|#[\\S]+'

_REMOVE_MISSED_UNICODE = '\[\?\]'

_SENTENCE_DELIMS = '\.\?\!;'

_REMOVE_PUNCTUATION = '"'.format(_SENTENCE_DELIMS)

_RE_PREPROC = re.compile('|'.join((_REMOVE_LINKS,
                                   _REMOVE_USER_HASH,
                                   _REMOVE_MISSED_UNICODE,
                                   _REMOVE_PUNCTUATION)))

_RE_SENTENCE = re.compile(('\\s*[{}]{{2,}}\\s*'
                           '|\\b\\s*[{}]\\s*').format(_SENTENCE_DELIMS,
                                                      _SENTENCE_DELIMS))

_HTMLPARSER = HTMLParser()

_RE_FOOD_POS = re.compile('^N.*|^JJ')

_FILTER_NO_RETWEET = lambda data: 'retweeted_status' not in data

_FILTER_EN = lambda data: '"lang":"en"' in data

_FILTER_POS = lambda (_, pos): _RE_FOOD_POS.match(pos) is not None

_GET_TEXT = lambda data: unidecode(_HTMLPARSER.unescape(data['text']))


def _build_noun_chunker():
    """ Build a noun chunker. """
    det_pos = "(<DT|PRP\$?|CD>|<DT>?<NN.?><POS>)"
    np_chunk = "{{{}?<JJ|W.*>*<NN.*>+}}".format(det_pos)
    np_grammar = "NP: {}".format(np_chunk)
    return RegexpParser(np_grammar)


def read_json(data_path):
    """ Read json tweet data ignoring retweets.  """
    with bz2.BZ2File(data_path, 'rb') as data:
        return [json.loads(line) for line in data
                if _FILTER_NO_RETWEET(line) and _FILTER_EN(line)]


def pos_tag_clean_text_data(data_json):
    """
    Remove hyperlinks and unprintable tokens from tweets, then tokenize and POS
    tag them.

    :param list data_json: list of json source data
    :return tuple: tuple of tuples of tokenized and POS tagged Tweets where
    each inner tuple is a pos tagged sentence from the source Tweet
    """
    return [
        tuple(
            pos_tag(word_tokenize(sentence))
            for sentence in
            it.imap(lambda s: s.strip(),
                    _RE_SENTENCE.split(_RE_PREPROC.sub('', text)))
            if len(sentence) > 0
        )
        for text in it.imap(_GET_TEXT, data_json)
    ]


def chunk_parse(pos_tagged_tweets):
    """
    Use a chunk parser to parse the output of pos_tag_clean_text_data().

    Each tuple is a Tweet and each nested tuple is the tokenized text of
    the source Tweet.

    :return list: chunker parsed tweets in same nested structure as input
    """
    chunker = _build_noun_chunker()
    return tuple(
        tuple(chunker.parse(sentence)
              for sentence in tweet)
        for tweet in pos_tagged_tweets)


_STATE_SCAN_EAT, \
    _STATE_EAT_LAST, \
    _STATE_NP_FOUND, \
    _STATE_IN_FOUND, \
    _STATE_NP_COMPLETE = range(5)


def parse_food_phrase(tree, eat_lexicon, filters, debug=False):
    """
    Extract the noun phrase after an eating verb. The lexicon of eating verbs
    must be given.

    The filters should remove vulgarities and other common stopwords in the
    corpus.
    """

    eat_lexicon_lower = set(tok.lower() for tok in eat_lexicon)
    state = _STATE_SCAN_EAT

    # Append a state to transition to complete at end of sentence.
    for stree in it.chain(tree, (('', ''),)):

        if state == _STATE_SCAN_EAT:

            if isinstance(stree, tuple) \
                    and stree[0].lower() in eat_lexicon_lower:
                state = _STATE_EAT_LAST

            assert state in [_STATE_SCAN_EAT, _STATE_EAT_LAST]

        elif state == _STATE_EAT_LAST:

            # We must have a noun phrase after our eat word.
            if not isinstance(stree, Tree) or stree.node != 'NP':
                state = _STATE_SCAN_EAT

            else:
                # Extract food from NP after eat word.
                new_words = [(w.lower(), pos) for w, pos in stree]
                words = new_words

                filtered_words = [(w, pos) for w, pos
                                  in it.ifilter(_FILTER_POS, new_words)
                                  if all(f(w) for f in filters)]

                if len(filtered_words) > 0:
                    state = _STATE_NP_FOUND
                else:
                    state = _STATE_SCAN_EAT

            assert state in [_STATE_SCAN_EAT, _STATE_NP_FOUND]

        elif state == _STATE_NP_FOUND:

            if isinstance(stree, tuple) \
                    and stree[0].lower() in ['of', 'in']:
                words.append(stree)
                filtered_words.append(stree)
                state = _STATE_IN_FOUND

            else:
                state = _STATE_NP_COMPLETE

            assert state in [_STATE_IN_FOUND, _STATE_NP_COMPLETE]

        elif state == _STATE_IN_FOUND:

            if not isinstance(stree, Tree) or stree.node != 'NP':
                words.pop()
                filtered_words.pop()

            else:
                new_words = [(w.lower(), pos) for w, pos in stree]
                words.extend(new_words)

                filtered_words.extend((w, pos) for w, pos
                                      in it.ifilter(_FILTER_POS, new_words)
                                      if all(f(w) for f in filters))

            state = _STATE_NP_COMPLETE

        elif state == _STATE_NP_COMPLETE:
            break

    # At the end of the sentence, check complete.

    if state == _STATE_NP_COMPLETE:

        food = ' '.join(w for w, _ in filtered_words)

        if debug is True and len(filtered_words) != len(words):
            food_unfiltered = ' '.join(w for w, _ in words)
            print "Filtered: {} => {}".format(food_unfiltered, food)

        if len(food) > 0:
            return food

    return None


def filters_from_dict(fdict):
    """
    Create set of infix, prefix, suffix, and match filters.

    N.B. the filter_dict must contains all of these keys.
    """
    return [lambda w: not any(fw in w for fw in fdict['infix']),
            lambda w: not any(len(w) > len(fw) and
                              fw == w[:len(fw)] for fw in fdict['prefix']),
            lambda w: not any(len(w) > len(fw) and
                              fw == w[-len(fw):] for fw in fdict['suffix']),
            lambda w: not any(fw == w for fw in fdict['match'])]


def count_foods(chunked_tweets, eat_lexicon, filters, debug=False):
    """ Count foods from pos tagged sentences using parse_food_phrase(). """
    counts = Counter(parse_food_phrase(tree, eat_lexicon, filters, debug)
                     for tweet in chunked_tweets
                     for tree in tweet)
    if None in counts:
        del counts[None]
    return counts
