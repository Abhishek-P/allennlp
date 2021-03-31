from typing import Iterable, Optional

from allennlp.data import DatasetReader, Token
from allennlp.data.fields import TextField, LabelField, ListField
from allennlp.data.instance import Instance
from datasets import load_dataset
from datasets.features import ClassLabel, Sequence, Translation, TranslationVariableLanguages
from datasets.features import Value


class HuggingfaceDatasetSplitReader(DatasetReader):
    """
        This reader implementation wraps the huggingface datasets package to utilize it's dataset management functionality
        and load the information in AllenNLP friendly formats
        Note: Reader works w.r.t to only one split of the dataset, i.e. you would need to create seperate reader for seperate splits

        Following dataset and configurations have been verified and work with this reader

                Dataset                       Dataset Configuration
                `xnli`                        `ar`
                `xnli`                        `en`
                `xnli`                        `de`
                `xnli`                        `all_languages`
                `glue`                        `cola`
                `glue`                        `mrpc`
                `glue`                        `sst2`
                `glue`                        `qqp`
                `glue`                        `mnli`
                `glue`                        `mnli_matched`
                `universal_dependencies`      `en_lines`
                `universal_dependencies`      `ko_kaist`
                `universal_dependencies`      `af_afribooms`
                `afrikaans_ner_corpus`        `NA`
                `swahili`                     `NA`
                `conll2003`                   `NA`
                `dbpedia_14`                  `NA`
                `trec`                        `NA`
                `emotion`                     `NA`
        """

    def __init__(
            self,
            max_instances: Optional[int] = None,
            manual_distributed_sharding: bool = False,
            manual_multiprocess_sharding: bool = False,
            serialization_dir: Optional[str] = None,
            dataset_name: [str] = None,
            split: str = 'train',
            config_name: Optional[str] = None,
    ) -> None:
        super().__init__(max_instances, manual_distributed_sharding, manual_multiprocess_sharding, serialization_dir)

        # It would be cleaner to create a separate reader object for different dataset
        self.dataset = None
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.index = -1

        if config_name:
            self.dataset = load_dataset(self.dataset_name, self.config_name, split=split)
        else:
            self.dataset = load_dataset(self.dataset_name, split=split)

    def _read(self, file_path) -> Iterable[Instance]:
        """
        Reads the dataset and converts the entry to AllenNLP friendly instance
        """
        for entry in self.dataset:
            yield self.text_to_instance(entry)

    def text_to_instance(self, *inputs) -> Instance:
        """
        Takes care of converting dataset entry into AllenNLP friendly instance
        Currently it is implemented in an unseemly catch-up model where it converts datasets.features that are required
        for the supported dataset, ideally it would require design where we cleanly map dataset.feature to an AllenNLP model
        and then go ahead with converting it one by one
        Doing that would provide the best chance of providing largest possible coverage with datasets

        Currently this is how datasets.features types are mapped to AllenNLP Fields

        dataset.feature type        allennlp.data.fields
        `ClassLabel`                  `LabelField` in feature name namespace
        `Value.string`                `TextField` with value as Token
        `Value.*`                     `LabelField` with value being label in feature name namespace
        `Sequence.string`             `ListField` of `TextField` with individual string as token
        `Sequence.ClassLabel`         `ListField` of `ClassLabel` in feature name namespace
        `Translation`                 `ListField` of 2 ListField (ClassLabel and TextField)
        `TranslationVariableLanguages`                 `ListField` of 2 ListField (ClassLabel and TextField)
        """

        # features indicate the different information available in each entry from dataset
        # feature types decide what type of information they are
        # e.g. In a Sentiment an entry could have one feature indicating the text and another indica
        features = self.dataset.features
        fields = dict()

        # TODO we need to support all different datasets features of https://huggingface.co/docs/datasets/features.html
        for feature in features:
            value = features[feature]

            # datasets ClassLabel maps to LabelField
            if isinstance(value, ClassLabel):
                field = LabelField(inputs[0][feature], label_namespace=feature, skip_indexing=True)

            # datasets Value can be of different types
            elif isinstance(value, Value):

                # String value maps to TextField
                if value.dtype == 'string':
                    # Since TextField has to be made of Tokens add whole text as a token
                    # TODO Should we use simple heuristics to identify what is token and what is not?
                    field = TextField([Token(inputs[0][feature])])

                else:
                    field = LabelField(inputs[0][feature], label_namespace=feature, skip_indexing=True)


            elif isinstance(value, Sequence):
                # datasets Sequence of strings to ListField of TextField
                if value.feature.dtype == 'string':
                    field_list = list()
                    for item in inputs[0][feature]:
                        item_field = TextField([Token(item)])
                        field_list.append(item_field)
                    if len(field_list) == 0:
                        continue
                    field = ListField(field_list)

                # datasets Sequence of strings to ListField of LabelField
                elif isinstance(value.feature, ClassLabel):
                    field_list = list()
                    for item in inputs[0][feature]:
                        item_field = LabelField(label=item, label_namespace=feature, skip_indexing=True)
                        field_list.append(item_field)
                    if len(field_list) == 0:
                        continue
                    field = ListField(field_list)

            # datasets Translation cannot be mapped directly but it's dict structure can be mapped to a ListField of 2 ListField
            elif isinstance(value, Translation):
                if value.dtype == "dict":
                    input_dict = inputs[0][feature]
                    langs = list(input_dict.keys())
                    field_langs = [LabelField(lang, label_namespace="languages") for lang in langs]
                    langs_field = ListField(field_langs)
                    texts = list()
                    for lang in langs:
                        texts.append(TextField([Token(input_dict[lang])]))
                    field = ListField([langs_field, ListField(texts)])

            # TranslationVariableLanguages is functionally a pair of Lists and hence mapped to a ListField of 2 ListField
            elif isinstance(value, TranslationVariableLanguages):
                # Although it is indicated as dict made up of a pair of lists
                if value.dtype == "dict":
                    input_dict = inputs[0][feature]
                    langs = input_dict["language"]
                    field_langs = [LabelField(lang, label_namespace="languages") for lang in langs]
                    langs_field = ListField(field_langs)
                    texts = list()
                    for lang in langs:
                        index = langs.index(lang)
                        texts.append(TextField([Token(input_dict["translation"][index])]))
                    field = ListField([langs_field, ListField(texts)])

            else:
                raise ValueError(f"Datasets feature type {type(value)} is not supported yet.")

            fields[feature] = field

        return Instance(fields)
