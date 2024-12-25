import os
from transformers import AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
from sentence_transformers import SentenceTransformer


def bert(cfg):
    bert_model_name = 'intfloat/multilingual-e5-large-instruct'
    # bert_model_name = 'dunzhang/stella_en_400M_v5'
    # bert_model_name = 'Alibaba-NLP/gte-base-en-v1.5'
    cache_dir = os.path.join('output', 'cache')
    cache_tokenizer_path = os.path.join(cache_dir, bert_model_name, 'tokenizer')
    cache_config_path = os.path.join(cache_dir, bert_model_name, 'config')
    cache_model_path = os.path.join(cache_dir, bert_model_name, 'model')
    if not os.path.exists(os.path.join(cache_dir, bert_model_name)):
        local_files_only = False
    else:
        local_files_only = True
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name, trust_remote_code=True,
                                              cache_dir=cache_tokenizer_path, local_files_only=local_files_only)
    config = AutoConfig.from_pretrained(bert_model_name, trust_remote_code=True,
                                        cache_dir=cache_config_path, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(bert_model_name, trust_remote_code=True,
                                      cache_dir=cache_model_path, config=config, local_files_only=local_files_only)
    cfg['target_size'] = tokenizer.vocab_size
    return model, tokenizer
