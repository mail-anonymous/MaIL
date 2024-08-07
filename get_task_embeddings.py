from transformers import AutoModel, AutoTokenizer
from hydra.utils import to_absolute_path
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf
import pickle


def get_task_embs(cfg, descriptions):

    if cfg.task_embedding_format == "one-hot":
        # offset defaults to 1, if we have pretrained another model, this offset
        # starts from the pretrained number of tasks + 1
        offset = cfg.task_embedding_one_hot_offset
        descriptions = [f"Task {i+offset}" for i in range(len(descriptions))]

    if cfg.task_embedding_format == "bert" or cfg.task_embedding_format == "one-hot":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif cfg.task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif cfg.task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif cfg.task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    return task_embs


def create_cfg_for_libero(task_embedding_format):
    cfg = DictConfig({'task_embedding_format': task_embedding_format,
                           'data': {'max_word_len': 25}})

    cfg.policy = OmegaConf.create()
    cfg.policy.language_encoder = OmegaConf.create()
    cfg.policy.language_encoder.network_kwargs = OmegaConf.create()

    return cfg


for task in ["libero_object", "libero_spatial", "libero_10", "libero_goal", "libero_90"]:

    # get task embedding
    task_suite = benchmark.get_benchmark_dict()[task]()

    cfg = create_cfg_for_libero("clip")

    tasks = {}
    if task != "libero_90":
        task_names = [task_suite.get_task(i).name for i in range(10)]
        descriptions = [task_suite.get_task(i).language for i in range(10)]
    else:
        task_names = [task_suite.get_task(i).name for i in range(90)]
        descriptions = [task_suite.get_task(i).language for i in range(90)]

    task_embs = get_task_embs(cfg, descriptions)

    for num, name in enumerate(task_names):
        tasks[name] = task_embs[num:num+1]

    with open("task_embeddings/" + task + ".pkl", 'wb') as f:  # open a text file
        pickle.dump(tasks, f) # serialize the list