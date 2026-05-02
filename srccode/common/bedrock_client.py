"""AWS Bedrock client using the unified `converse` API.

Why `converse`?
  * Works across all Bedrock model families (Anthropic, Meta, Mistral,
    Amazon Titan/Nova, Cohere) with the same payload shape.
  * Returns input/output token usage in `response['usage']`.
  * Honours Bedrock's standard inference parameters
    (`maxTokens`, `temperature`, `topP`, `stopSequences`).

Authentication follows the standard AWS credential chain:
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and optional AWS_SESSION_TOKEN)
    or AWS_PROFILE pointing at ~/.aws/credentials,
    or an instance profile if running on EC2/ECS/Lambda.

Region & credentials follow the standard AWS resolution chain (env vars,
`aws configure`, AWS_PROFILE, SSO, instance role).
"""
from __future__ import annotations

import threading
from typing import Optional

from . import config

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                try:
                    import boto3  # type: ignore
                except ImportError as e:
                    raise RuntimeError(
                        "boto3 is required for --provider=bedrock. "
                        "Install with: pip install boto3"
                    ) from e
                kwargs = {}
                if config.AWS_REGION:
                    kwargs["region_name"] = config.AWS_REGION
                if config.BEDROCK_PROFILE:
                    session = boto3.Session(profile_name=config.BEDROCK_PROFILE)
                    _client = session.client("bedrock-runtime", **kwargs)
                else:
                    _client = boto3.client("bedrock-runtime", **kwargs)
    return _client


def chat(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    seed: int = 42,            # accepted for signature parity; Bedrock has no seed param on most models
    num_ctx: Optional[int] = None,   # accepted for parity; Bedrock context is fixed by the model
    num_predict: Optional[int] = 4096,
) -> tuple[str, dict]:
    """Single-turn chat against Bedrock via converse().

    Returns:
        (assistant_text, usage_dict) with input_tokens/output_tokens/total_tokens.
    """
    del seed, num_ctx  # unused on Bedrock; kept for API compatibility

    client = _get_client()
    inference_config = {"temperature": float(temperature)}
    if num_predict is not None:
        inference_config["maxTokens"] = int(num_predict)

    resp = client.converse(
        modelId=model,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig=inference_config,
    )

    # Extract assistant text from the converse output.
    out_msg = resp.get("output", {}).get("message", {})
    parts = out_msg.get("content", []) or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))

    usage_raw = resp.get("usage", {}) or {}
    in_tok = int(usage_raw.get("inputTokens", 0) or 0)
    out_tok = int(usage_raw.get("outputTokens", 0) or 0)
    usage = {
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "total_tokens":  int(usage_raw.get("totalTokens", in_tok + out_tok) or 0),
    }
    return text, usage


def health_check() -> str:
    """Verify boto3 is importable and the client can be constructed.

    We do *not* call ListFoundationModels (different IAM permissions) — a
    successful `client(...)` is enough to confirm credentials resolved.
    """
    try:
        _get_client()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Cannot initialise AWS Bedrock client: {e}") from e
    profile = f"  profile={config.BEDROCK_PROFILE}" if config.BEDROCK_PROFILE else ""
    region = _get_client().meta.region_name or "<unset>"
    if config.BEDROCK_API_KEY:
        auth = "  auth=api-key"
    elif config.BEDROCK_PROFILE:
        auth = ""  # already shown via profile
    else:
        auth = "  auth=aws-cli-chain"
    return f"OK: AWS Bedrock  region={region}{profile}{auth}"
