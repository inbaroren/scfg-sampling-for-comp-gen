import comet_ml
import csv
from typing import Dict, Optional, Union
import logging
import copy
import pandas as pd
import numpy as np

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from models_code.data.tokenizers.thingtalk_tokenizer import DOM_TO_TOK, convert_to_lower

logger = logging.getLogger(__name__)


@DatasetReader.register("my_seq2seq_pd")
class Seq2SeqPdDatasetReader(DatasetReader):
    """
    Reads examples from a tsv file using pandas

    # Parameters

    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    target_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define output (target side) token representations. Defaults to
        `source_token_indexers`.
    source_add_start_token : `bool`, (optional, default=`True`)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    source_add_end_token : `bool`, (optional, default=`True`)
        Whether or not to add `END_SYMBOL` to the end of the source sequence.
    delimiter : `str`, (optional, default=`"\t"`)
        Set delimiter for tsv/csv file.
    quoting : `int`, (optional, default=`csv.QUOTE_MINIMAL`)
        Quoting to use for csv reader.
    debug: if True, only 100 lines are loaded
    example_id_col: name of the column with the examples id (optional, default=0)
    utterance_col: name of the column with the utterances (input) (optional, default=1)
    program_col: name of the column with the programs (output) (optional, default=2)
    condition_name: if the training file has data for several experiments, this is the name of the current experiment.
        when reading a file, it is expected to have a column <condition_name>, where 1 denotes training examples
        and 2 validation examples. (optional, default=None)
    condition_val: used to differentiate between training and test examples by conditioning `condition_name` to be
        equal to `condition_val` (optional, default=None)
    read_header: None or "infer". if None, the header is ignored, so utterance_col, program_col and condition_name must
        be numbers. (optional, default=None)
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        source_add_end_token: bool = True,
        target_add_start_token: bool = True,
        target_add_end_token: bool = True,
        start_symbol: str = START_SYMBOL,
        end_symbol: str = END_SYMBOL,
        delimiter: str = "\t",
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        quoting: int = csv.QUOTE_MINIMAL,
        debug: bool = False,
        example_id_col: Union[int, str] = 0,
        utterance_col: Union[int, str] = 1,
        program_col: Union[int, str] = 2,
        condition_name: str = None,
        condition_value: int = None,
        read_header: str = None,
        lower_case_output: bool = False,
        add_domain_token: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._utterance_col = utterance_col
        self._program_col = program_col
        self._example_id_col = example_id_col

        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers

        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token
        self._start_token: Optional[Token] = None
        self._end_token: Optional[Token] = None
        if (
            source_add_start_token
            or source_add_end_token
            or target_add_start_token
            or target_add_end_token
        ):
            try:
                self._start_token, self._end_token = self._source_tokenizer.tokenize(
                    start_symbol + " " + end_symbol
                )
            except ValueError:
                raise ValueError(
                    f"Bad start or end symbol ({'start_symbol', 'end_symbol'}) "
                    f"for tokenizer {self._source_tokenizer}"
                )

        if add_domain_token:
            special_tokens_dict = {'additional_special_tokens': list(DOM_TO_TOK.values())}
            self._source_tokenizer.tokenizer.add_special_tokens(special_tokens_dict)
            self._source_token_indexers["tokens"]._allennlp_tokenizer.tokenizer.add_special_tokens(special_tokens_dict)

        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting
        self.debug = debug
        if (condition_name is None and condition_value is not None) or (condition_name is not None and condition_value is None):
            logger.warning("Condition should be composed of name and value (condition_name, condition_value), but one argument is missing")
        self._condition_name = condition_name
        self._condition_val = condition_value
        self._read_header = read_header
        self._lower_case_output = lower_case_output
        self._add_domain_token = add_domain_token

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

        logger.info("Reading instances from lines in file at: %s", file_path)
        df = pd.read_csv(file_path, sep='\t', header=self._read_header)
        # filter train/dev examples if needed
        if self._condition_name is not None:
            df = df[df[self._condition_name] == self._condition_val]
        # keep only the input/output data
        df = df[[self._example_id_col, self._utterance_col, self._program_col]]

        for line_num in range(df.shape[0]):
            if self.debug and line_num > 1000:
                break

            row = df.iloc[line_num]
            if len(row) < 2:
                logger.info("Invalid line format: %s (line number %d)" % (row, line_num + 1))
                raise ConfigurationError(
                    "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                )
            if len(row) != 3:
                logger.info("Evaluation, no target")

            example_id, source_sequence, target_sequence = row[self._example_id_col], row[self._utterance_col], row[self._program_col]
            if len(source_sequence) == 0:
                continue
            if isinstance(target_sequence, np.float):
                target_sequence = None
            yield self.text_to_instance(example_id, source_sequence, target_sequence)
        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    @overrides
    def text_to_instance(
        self, example_id: str, source_string: str, target_string: str = None
    ) -> Instance:  # type: ignore
        metadata_field = MetadataField({'example_id': example_id})
        source_string = source_string.lower() if self._lower_case_output else source_string
        if self._add_domain_token:
            domain = example_id.split('-')[0]
            domain_special_token = DOM_TO_TOK[domain]
            source_string = domain_special_token + " " + source_string
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]
        if self._source_add_start_token:
            tokenized_source.insert(0, copy.deepcopy(self._start_token))
        if self._source_add_end_token:
            tokenized_source.append(copy.deepcopy(self._end_token))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            target_string = convert_to_lower(target_string) if self._lower_case_output else target_string
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]
            if self._target_add_start_token:
                tokenized_target.insert(0, copy.deepcopy(self._start_token))
            if self._target_add_end_token:
                tokenized_target.append(copy.deepcopy(self._end_token))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field, "metadata": metadata_field})
        else:
            return Instance({"source_tokens": source_field, "metadata": metadata_field})


