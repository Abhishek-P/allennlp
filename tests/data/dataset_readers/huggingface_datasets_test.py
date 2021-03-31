import pytest

from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.hugging_face_datasets_reader import HuggingfaceDatasetSplitReader
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
import logging

logger = logging.getLogger(__name__)

# TODO these UTs are actually downloading the datasets and will be very very slow
class HuggingfaceDatasetSplitReaderTest:


    SUPPORTED_DATASETS_WITHOUT_CONFIG = ["afrikaans_ner_corpus", "dbpedia_14", "trec", "swahili", "conll2003", "emotion"]

    """
        Running the tests for supported datasets which do not require config name to be specified
    """
    @pytest.mark.parametrize("dataset", SUPPORTED_DATASETS_WITHOUT_CONFIG)
    def test_read_for_datasets_without_config(self, dataset):
        huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset)
        instances = list(huggingface_reader.read(None))
        assert len(instances) == len(huggingface_reader.dataset)

    # Not testing for all configurations only some
    SUPPORTED_DATASET_CONFIGURATION = (
        ("glue", "cola"),
        ("universal_dependencies", "af_afribooms"),
        ("xnli", "all_languages")
    )

    """
        Running the tests for supported datasets which require config name to be specified
    """
    @pytest.mark.parametrize("dataset, config", SUPPORTED_DATASET_CONFIGURATION)
    def test_read_for_datasets_requiring_config(self, dataset, config):
        huggingface_reader = HuggingfaceDatasetSplitReader(dataset_name=dataset, config_name=config)
        instances = list(huggingface_reader.read(None))
        assert len(instances) == len(huggingface_reader.dataset)




