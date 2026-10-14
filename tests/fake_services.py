"""Test-only SDK endpoints. No network or model publication is performed."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


class Services:
    def __init__(self, cache):
        self.cache = cache
        self.calls = []
        self.uploads = {}
        self.network_error = False
        self.missing_hash = False
        self.card_error = False
        self.card = NS(data=None, text="ORIGINAL MODEL CARD\n", push_to_hub=self.card_push)

    def card_push(self, *args, **kwargs):
        self.calls.append(("card", args, kwargs))

    def push(self, *args, **kwargs):
        self.calls.append(("weights", args, kwargs))

    def model_info(self, *args, **kwargs):
        self.calls.append(("model_info", args, kwargs))
        return NS(sha="base-commit", siblings=[
            NS(rfilename="model.safetensors", lfs={} if self.missing_hash else {"sha256": "abc123"}),
            NS(rfilename="config.json", lfs=None),
        ])

    def dataset_info(self, *args, **kwargs):
        return NS(sha="dataset-commit")

    def load_card(self, *args, **kwargs):
        if self.card_error:
            raise ConnectionError("model card endpoint unavailable")
        self.calls.append(("load_card", args, kwargs))
        return self.card

    def upload_file(self, **kwargs):
        self.uploads[kwargs["path_in_repo"]] = Path(kwargs["path_or_fileobj"]).read_text()
        self.calls.append(("upload_file", (), kwargs))

    def list_models(self, **kwargs):
        return iter([
            NS(id="owner/quantized", tags=["gguf"], gated=False),
            NS(id="owner/gated", tags=[], gated=True),
            NS(id="owner/missing", tags=None, gated=False),
            NS(id="owner/model", tags=[], gated=False),
        ])

    def auth_check(self, *args, **kwargs):
        if self.network_error:
            raise ConnectionError("network unavailable")
        from huggingface_hub.utils import GatedRepoError
        import httpx
        raise GatedRepoError("access denied", response=httpx.Response(403, request=httpx.Request("GET", "https://fixture.invalid/model")))

    def get_paths_info(self, repo, *args, **kwargs):
        return [] if repo.endswith("missing") else [NS(last_commit=NS(oid="abcdef012345"))]

    def download(self, *args, **kwargs):
        self.calls.append(("download", args, kwargs))
        return self.cache

    def __enter__(self):
        import huggingface_hub as hub
        import transformers
        import peft
        self.stack = ExitStack()
        api = NS(model_info=self.model_info, list_models=self.list_models,
                 auth_check=self.auth_check, get_paths_info=self.get_paths_info)
        for target, name, value in [
            (hub, "HfApi", lambda: api),
            (hub, "model_info", self.model_info),
            (hub, "dataset_info", self.dataset_info),
            (hub, "upload_file", self.upload_file),
            (hub, "hf_hub_download", self.download),
            (hub.ModelCard, "load", self.load_card),
            (transformers.PreTrainedModel, "push_to_hub", self.push),
            (peft.PeftModel, "push_to_hub", self.push),
            (transformers.PreTrainedTokenizerBase, "push_to_hub", self.push),
            (transformers.ProcessorMixin, "push_to_hub", self.push),
        ]:
            self.stack.enter_context(patch.object(target, name, value))
        return self

    def __exit__(self, *args):
        self.stack.close()


class Accelerators:
    def __init__(self, kind, original_system=None):
        self.kind = kind
        self.original = original_system

    def __enter__(self):
        import torch
        self.stack = ExitStack()
        for backend in ["cuda", "xpu", "mlu", "sdaa", "musa", "npu"]:
            active = self.kind == backend or (backend == "cuda" and self.kind == "rocm")
            device = NS(is_available=lambda active=active: active, device_count=lambda: 2,
                        get_device_name=lambda i: f"Device {i}",
                        get_device_properties=lambda i: NS(total_memory=8 * 1024**3),
                        mem_get_info=lambda i: (7 * 1024**3, 8 * 1024**3),
                        empty_cache=lambda: None)
            self.stack.enter_context(patch.object(torch, backend, device, create=True))
        self.stack.enter_context(patch.object(torch.version, "hip", "6.0" if self.kind == "rocm" else None))
        self.stack.enter_context(patch.object(torch.version, "cuda", "13.0"))
        self.stack.enter_context(patch.object(torch.version, "cann", "8.0", create=True))
        self.stack.enter_context(patch.object(torch.backends.mps, "is_available", lambda: self.kind == "mps"))
        self.stack.enter_context(patch.object(torch.mps, "empty_cache", lambda: None))
        if self.original is not None:
            for backend in ["xpu", "mlu", "sdaa", "musa", "npu"]:
                self.stack.enter_context(patch.object(self.original, f"is_{backend}_available", lambda backend=backend: self.kind == backend))
        return self

    def __exit__(self, *args):
        self.stack.close()


class Benchmarks:
    """Evaluate a local task using the real lm-eval engine, without remote datasets."""
    def __init__(self, directory):
        self.directory = directory

    def __enter__(self):
        import json
        import lm_eval
        from lm_eval.tasks import TaskManager
        root = Path(self.directory)
        rows = [dict(prompt="Hello", label=0), dict(prompt="Good morning", label=1), dict(prompt="The sky", label=0)]
        (root / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (root / "fixture.yaml").write_text(
            "task: heretic_fixture\n"
            "dataset_path: json\n"
            "dataset_kwargs:\n"
            f"  data_files: {root / 'data.jsonl'}\n"
            "test_split: train\n"
            "output_type: multiple_choice\n"
            "doc_to_text: '{{prompt}}'\n"
            "doc_to_choice: ['world', 'red']\n"
            "doc_to_target: '{{label}}'\n"
            "metric_list:\n"
            "  - metric: acc\n"
            "    aggregation: mean\n"
            "    higher_is_better: true\n"
        )
        actual = lm_eval.simple_evaluate
        manager = TaskManager(include_path=root)

        def evaluate(*args, **kwargs):
            kwargs.update(task_manager=manager, bootstrap_iters=0, log_samples=False)
            return actual(*args, **kwargs)

        self.context = patch.object(lm_eval, "simple_evaluate", evaluate)
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        self.context.__exit__(*args)
