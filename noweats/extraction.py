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

_REMOVE_USER_HASH = '^@[\\S]+\\s*|@\\b|#\\b'

_REMOVE_MISSED_UNICODE = '\[\?\]'

_REMOVE_PUNCTUATION = '"'

_RE_PREPROC = re.compile('|'.join((_REMOVE_LINKS,
                                   _REMOVE_USER_HASH,
                                   _REMOVE_MISSED_UNICODE,
                                   _REMOVE_PUNCTUATION)))

_SENTENCE_DELIMS = '\.\?\!;\n'

_RE_SENTENCE = re.compile(('\\s*[{}]{{2,}}\\s*'
                           '|\\b\\s*[{}]\\s*').format(_SENTENCE_DELIMS,
                                                      _SENTENCE_DELIMS))

_FIX_WHITESPACE = re.compile('[\\s\\\/]+')
_HTMLPARSER = HTMLParser()

_RE_FOOD_POS = re.compile('^N.*|^JJ')

_RE_TWEET_AND = re.compile('"lang"\\s*:\\s*"en"')
_RE_TWEET_NOT = re.compile('"retweeted_status"\\s*:'
                           '|"text"\\s*:\\s*"\\s*RT'
                           '|"lang"\\s*:\\s*"(?!en)')

_FILTER_TWEET = lambda datas: _RE_TWEET_AND.search(datas) is not None \
    and _RE_TWEET_NOT.search(datas) is None

_FILTER_POS = lambda (_, pos): _RE_FOOD_POS.match(pos) is not None

_GET_TEXT = lambda data: unidecode(_HTMLPARSER.unescape(data['text']))


def _build_noun_chunker():
    """ Build a noun chunker. """
    det_pos = "(<DT|PRP\$?|CD>|<DT>?<NN.?><POS>)"
    np_chunk = "{{{}?<JJ|W.*>*<NN.*>+}}".format(det_pos)
    np_grammar = "NP: {}".format(np_chunk)
    return RegexpParser(np_grammar)

_CHUNKER = _build_noun_chunker()


def read_json(data_path):
    """ Read json tweet data ignoring retweets.  """
    with bz2.BZ2File(data_path, 'rb') as data_file:
        return [json.loads(line)
                for line in data_file if _FILTER_TWEET(line)]


def sentence_split_clean_data(data_json, eat_lexicon):
    """
    Remove hyperlinks and unprintable tokens from tweets and split them into
    sentences.

    :param list data_json: list of json source data
    :param list eat_lexicon: list of eat words
    :return list: list of tuples of sentences split from tweets where each
    sentence contains at least one word from the eat lexicon
    """
    eat_lexicon_re = re.compile('|'.join(eat_lexicon), re.IGNORECASE)
    return [
        sentences for sentences in
        (tuple(_FIX_WHITESPACE.sub(' ', sentence) for sentence in
               it.imap(lambda s: s.strip(),
                       _RE_SENTENCE.split(_RE_PREPROC.sub('', text)))
               if len(sentence) > 0
               and eat_lexicon_re.search(sentence) is not None
               )
         for text in it.imap(_GET_TEXT, data_json)
         )
        if len(sentences) > 0
    ]


def score_tweet_en(tweet, en_model):
    """ Score a tweet. """
    scores = [s for s in it.imap(en_model, it.chain(*tweet)) if s is not None]
    if len(scores) > 0:
        return sum(scores) / float(len(scores))
    else:
        return 0


def tokenize_tweet(tweet):
    """ Tokenize a tweet. """
    return tuple(word_tokenize(sentence) for sentence in tweet)


def tokenize_keep_en_tweets(tweets, en_model, keep_pct=0.95):
    """
    Tokenize tweets split already into sentences.

    Use the prefix-suffix model of english to rank tweets and then discard some
    proportion.
    """
    by_score = sorted((tokenize_tweet(tweet) for tweet in tweets),
                      key=lambda tweet: score_tweet_en(tweet, en_model),
                      reverse=True)
    to_keep = int(keep_pct * len(by_score))
    return by_score[:to_keep]


def pos_tag_tweet(tweet):
    """ POS tag tweets split already into sentences. """
    return tuple(pos_tag(sentence) for sentence in tweet)


def chunk_tweet(pos_tagged_tweet):
    """
    Use a chunk parser to parse the output of pos_tag_clean_text_data().

    Each tuple is a Tweet and each nested tuple is the tokenized text of
    the source Tweet.

    :return list: chunker parsed tweets in same nested structure as input
    """
    return tuple(_CHUNKER.parse(sentence) for sentence in pos_tagged_tweet)


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
                    and (stree[0].lower() in ['of', 'in', 'on', 'with']
                         or stree[0].lower() in ['and']):
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


def make_en_prefix_suffix_model(model):
    """ Create a function that scores English words. """

    re_not_alpha, prefixes, suffixes, p_default = model

    def p_word(word):
        """ Compute probability of a word under the model. """

        if len(word) < 3 or re_not_alpha.match(word) is not None:
            return None

        prefix, suffix = word[:3].lower(), word[-3:].lower()

        p_prefix = prefixes[prefix] if prefix in prefixes else p_default
        p_suffix = suffixes[suffix] if suffix in suffixes else p_default
        return p_prefix * p_suffix

    return p_word


def build_en_prefix_suffix_model(data_json):
    """ Create a probabilisitc model of English words from data. """

    re_not_alpha = re.compile('^[^a-zA-Z]+$')
    all_tweets = sentence_split_clean_data(data_json, ['.'])
    toks = [t for tw in all_tweets
            for s in tw
            for t in word_tokenize(s)]
    prefixes = Counter(t[:3].lower() for t in toks
                       if len(t) > 2 and re_not_alpha.match(t) is None)
    suffixes = Counter(t[-3:].lower() for t in toks
                       if len(t) > 2 and re_not_alpha.match(t) is None)
    # There are 26 alpha characters, 1 apostrophe, and 10 digits.
    total_cmb = 37**3
    total_words = sum(prefixes.itervalues())
    assert total_words == sum(suffixes.itervalues()), "Bad words count"
    norm = float(total_cmb + total_words)

    # Laplace smooth probabilities using total_cmb (adding one to every
    # combination).
    for k in prefixes:
        prefixes[k] /= norm
    for k in suffixes:
        suffixes[k] /= norm
    p_default = 1. / norm

    return [re_not_alpha, prefixes, suffixes, p_default]
