"""
Extract foods that people are eating from their Tweets.
"""
from nltk.chunk import RegexpParser
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, Tree
from noweats.util import counter
from unidecode import unidecode
from HTMLParser import HTMLParser

import re
import json
import itertools as its
import bz2

_REMOVE_LINKS = '\\s?\\bhttp[\S]+'

_REMOVE_MISSED_UNICODE = '\[\?\]'

_REMOVE_USER_HASH = '|'.join((
    '@[\\S]+',                   # any @mention
    '(#[\\S]+\\s+){1,}#[\\S]+',  # 2 or more #hashtags in a row
    '#\\b',                      # remove # from single #hashtag
))

_REMOVE_SPACED_CHARS = '(\\A|\\s)\\w(\\s+\\w){2,}(?=\\Z|\\s)'

_RE_PREPROC = re.compile('|'.join((_REMOVE_LINKS,
                                   _REMOVE_MISSED_UNICODE,
                                   _REMOVE_USER_HASH,
                                   _REMOVE_SPACED_CHARS,
                                   ))
                         )

_RE_SENTENCE = re.compile(
    '|'.join(('\\s*[\?\!;\n]{1,}\\s*',  # 1 or more non-period
              '\\s*\.{2,}\\s*',         # 2 or more periods
              '\\s*\.(?![0-9])\\s*',    # not a number
              )))

_RE_REMOVE_CHARS = re.compile('[^\\s\\w@&+()\',-]+')
_RE_FIX_WHITESPACE = re.compile('[\\s\\\/]+')

_HTMLPARSER = HTMLParser()

_RE_FOOD_POS = re.compile('^N.*|^JJ')

_RE_LANG_EN = re.compile('"lang"\\s*:\\s*"en"')
_RE_TWEET_NOT = re.compile('|'.join((
    '"retweeted_status"\\s*:',  # is a retweet
    '"text"\\s*:\\s*"\\s*RT',   # is a retweet by text
    '"lang"\\s*:\\s*"(?!en)',   # is non-English
)))

_FILTER_TWEET = lambda datas: _RE_LANG_EN.search(datas) is not None \
    and _RE_TWEET_NOT.search(datas) is None

_FILTER_POS = lambda (_, pos): _RE_FOOD_POS.match(pos) is not None

_GET_TEXT = lambda data: unidecode(_HTMLPARSER.unescape(data['text']))

_STOPWORDS_EN = set(stopwords.words('english'))


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
    # Note that test for eat_lexicon guarantees that the sentence will have
    # nonzero length.
    eat_lexicon_re = re.compile('|'.join(eat_lexicon), re.IGNORECASE)
    return [
        sentences for sentences in
        (tuple(_RE_FIX_WHITESPACE
               .sub(' ', _RE_REMOVE_CHARS.sub(' ', sentence)).strip()
               for sentence in _RE_SENTENCE.split(_RE_PREPROC.sub('', text))
               if eat_lexicon_re.search(sentence) is not None
               )
         for text in its.imap(_GET_TEXT, data_json)
         )
        if len(sentences) > 0
    ]


def remove_dups(tweets, keep_thresh=1):
    """ Filter duplicate tweets since they are likely to be spam. """
    hash_tw = lambda tw: tuple(set(w for w in (w.lower()
                                               for s in tw
                                               for w in s.split())
                                   if w not in _STOPWORDS_EN))
    hashes = counter(hash_tw(tw) for tw in tweets)
    return [tw for tw in tweets if hashes[hash_tw(tw)] <= keep_thresh]


def score_tweet_en(tweet, en_model):
    """ Score a tweet. """
    scores = [s for s in its.imap(en_model, its.chain(*tweet)) if s is not None]
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
    for stree in its.chain(tree, (('', ''),)):

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
                                  in its.ifilter(_FILTER_POS, new_words)
                                  if all(f(w) for f in filters)]

                if len(filtered_words) > 0:
                    state = _STATE_NP_FOUND
                else:
                    state = _STATE_SCAN_EAT

            assert state in [_STATE_SCAN_EAT, _STATE_NP_FOUND]

        elif state == _STATE_NP_FOUND:

            newstate = _STATE_NP_COMPLETE
            if isinstance(stree, tuple):
                word = stree[0].lower()
                if word in ['of', 'in', 'on', 'with', 'and']:
                    words.append(stree)
                    filtered_words.append((word, stree[1]))
                    newstate = _STATE_IN_FOUND

            state = newstate

            assert state in [_STATE_IN_FOUND, _STATE_NP_COMPLETE]

        elif state == _STATE_IN_FOUND:

            if not isinstance(stree, Tree) or stree.node != 'NP':
                words.pop()
                filtered_words.pop()

            else:
                new_words = [(w.lower(), pos) for w, pos in stree]
                words.extend(new_words)

                filtered_words.extend((w, pos) for w, pos
                                      in its.ifilter(_FILTER_POS, new_words)
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

        if len(food) > 0 and food not in _STOPWORDS_EN:
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
    counts = counter(parse_food_phrase(tree, eat_lexicon, filters, debug)
                     for tweet in chunked_tweets
                     for tree in tweet)
    if None in counts:
        del counts[None]
    return counts


def allowed_chars_no_whitespace():
    """ Compute set of chars that pass the filtering stage. """
    allchars = ''.join(chr(i) for i in range(256)).lower()
    return set(_RE_FIX_WHITESPACE.sub('', _RE_REMOVE_CHARS.sub(' ', allchars)))
