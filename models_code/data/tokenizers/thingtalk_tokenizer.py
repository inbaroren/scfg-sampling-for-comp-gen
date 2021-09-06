from typing import List
import re
from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer

DOM_TO_TOK = {"restaurants": '<rest>', "books": '<book>', "hotels": '<hotel>', "people": '<peopl>', "music": '<music>',
              "movies": '<movie>'}


def add_spaces(x):
    """
    Each meaningful token in the thingtalk syntax is a concatenation of smaller atoms
    For example: "param:homeLocation:Number" or "param:id:Entity(org.schema.Book:Book)" or "param:language:Entity(tt.iso_lang)"
    Tables: "@org.schema.Book:Book"
    Separating by ' ' creates a large vocabulary.
    """
    # add spaces after separators
    x = re.sub(r'([:.@\(\)\[\]",]{1})', ' \g<1> ', x)
    x = x.replace('^^', ' ^^ ')
    # remove multiple spaces
    x = re.sub(r'\s\s+', ' ', x)

    return x


def convert_to_lower(thingtalk_program: str) -> str:
    """
    when moving to lower case, add a "_" between each two words to maintain the meaning
    example:
        convert_to_lower(ratingValue)
        >> rating_value
    """
    new_program = []
    for tok in thingtalk_program.split():
        while re.findall(r'([a-z])([A-Z])', tok):
            tok = re.sub(r'([a-z])([A-Z])', '\g<1>_\g<2>', tok)
        new_program.append(tok.lower())
    return ' '.join(new_program)


@Tokenizer.register("thingtalk_with_bert")
class ThinkTalkBERTTokenizer(Tokenizer):
    """
    A `Tokenizer` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.

    Note that we use `text.split()`, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.

    Registered as a `Tokenizer` with name "whitespace" and "just_spaces".
    """
    def __init__(self, model_name):
        self._bert_tokenizer = PretrainedTransformerTokenizer(model_name, add_special_tokens=False)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        text = add_spaces(text)
        # tokenize the quoted texts with bert, keep ThingTalk tokens complete:|
        text_sections = [(0,0)]
        for m in re.finditer(r'" [0-9A-Za-z\'_\-.:;,?\s]+ "', text):
            text_sections.append((m.start(0)+1, m.end(0)-1))

        if not text_sections:
            return [Token(t) for t in text.split()]

        tokenized_text = []
        for i in range(1, len(text_sections)):
            prev_s = text_sections[i-1][1]
            s = text_sections[i][0]
            e = text_sections[i][1]
            tokenized_text.extend([Token(t) for t in text[prev_s:s].strip().split()])
            tokenized_text.extend(self._bert_tokenizer.tokenize(text[s:e].strip()))
        if text_sections[-1][1] < len(text):
            tokenized_text.extend([Token(t) for t in text[text_sections[-1][1]:].strip().split()])
        return tokenized_text


@Tokenizer.register("thingtalk_with_bart")
class ThinkTalkBARTTokenizer(Tokenizer):
    """
    A `Tokenizer` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.

    Note that we use `text.split()`, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.

    Registered as a `Tokenizer` with name "whitespace" and "just_spaces".
    """
    def __init__(self, model_name, add_special_tokens=True):
        self._bart_tokenizer = PretrainedTransformerTokenizer(model_name, add_special_tokens=False)
        self._add_special_tokens = add_special_tokens
        self._bos_token = self._bart_tokenizer.tokenizer._bos_token
        self._eos_token = self._bart_tokenizer.tokenizer._eos_token

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        text = add_spaces(text)
        # tokenize the quoted texts with bert, keep ThingTalk tokens complete:|
        text_sections = [(0,0)]
        for m in re.finditer(r'" [0-9A-Za-z\'_\-.:;,?\s]+ "', text):
            text_sections.append((m.start(0)+1, m.end(0)-1))

        if not text_sections:
            return [Token(self._bos_token)]+[Token(t) for t in text.split()]+[Token(self._eos_token)]

        tokenized_text = []
        for i in range(1, len(text_sections)):
            prev_s = text_sections[i-1][1]
            s = text_sections[i][0]
            e = text_sections[i][1]
            tokenized_text.extend([Token(t) for t in text[prev_s:s].strip().split()])
            tokenized_text.extend(self._bart_tokenizer.tokenize(text[s:e].strip()))
        if text_sections[-1][1] < len(text):
            tokenized_text.extend([Token(t) for t in text[text_sections[-1][1]:].strip().split()])

        tokenized_text.insert(0, self._bart_tokenizer.tokenize(self._bos_token)[0])
        tokenized_text.append(self._bart_tokenizer.tokenize(self._eos_token)[0])

        return tokenized_text

