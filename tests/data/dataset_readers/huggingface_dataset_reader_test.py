import pytest

from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.hugging_face_datasets_reader import HuggingfaceDatasetSplitReader
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
import logging
logger = logging.getLogger(__name__)
from datasets import list_datasets
class HuggingFaceDataSetReaderTest:

    @pytest.mark.parametrize("dataset, config", (("afrikaans_ner_corpus", None), ("dbpedia_14", None), ("universal_dependencies", "af_afribooms"), ("trec", None), ("swahili", None)))
    def test_read_for_datasets(self, dataset, config):
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_glue_with_config_cola(self):
        dataset = "glue"
        config = "cola"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_glue_with_config_mrpc(self):
        dataset = "glue"
        config = "mrpc"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_squad(self):
        dataset = "squad"
        config = None
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_conll2003(self):
        dataset = "conll2003"
        config = None
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_pubmed_qa_with_config_pqa_labeled(self):
        dataset = "pubmed_qa"
        config = "pqa_labeled"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_pubmed_qa_with_config_pqa_unlabeled(self):
        dataset = "pubmed_qa"
        config = "pqa_unlabeled"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_pubmed_qa_with_config_pqa_artificial(self):
        dataset = "pubmed_qa"
        config = "pqa_artificial"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_xnli_with_config_en(self):
        dataset = "xnli"
        config = "en"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_xnli_with_config_de(self):
        dataset = "xnli"
        config = "de"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_xnli_with_config_all_languages(self):
        dataset = "xnli"
        config = "all_languages"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_afrikaans_ner_corpus(self):
        dataset = "afrikaans_ner_corpus"
        config = None
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_dbpedia_14(self):
        dataset = "dbpedia_14"
        config = None
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_swahili(self):
        dataset = "swahili"
        config = None
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_universal_dependencies_with_config_en_lines(self):
        dataset = "universal_dependencies"
        config = "en_lines"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_universal_dependencies_with_config_ko_kaist(self):
        dataset = "universal_dependencies"
        config = "ko_kaist"
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_trec(self):
        dataset = "trec"
        config = None
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)


    def test_read_for_emotion(self):
        dataset = "emotion"
        config = None
        try:
            print("trying for dataset %s", dataset,  flush=True)
            if config:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
            else:
                huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
            instances = list(huggingface_reader.read(None))
            print(instances[0], huggingface_reader.dataset[0])
            print("instances", len(instances), "for", dataset, flush=True)
        except Exception as E:
            print("Failed", E, flush=True)
© 2021 GitHub, Inc.