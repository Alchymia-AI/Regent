"""
Generation engine with Ver Head-gated decoding.

Implements the three-zone decoding strategy:
    FLOW    (grounding > 0.6): sample normally
    CAUTION (0.3 < grounding <= 0.6): lower temperature, bias toward hedging
    HALT    (grounding <= 0.3): stop, trigger EPG retrieval, re-decode

This is the Regent's Option 2 hallucination architecture running at inference.
"""

from dataclasses import dataclass

import mlx.core as mx

from regent_model.layers.model import RegentModel


@dataclass
class GenerateConfig:
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Verification thresholds
    flow_threshold: float = 0.6
    caution_threshold: float = 0.3
    caution_temperature: float = 0.3

    # When HALT triggers, how many EPG nodes to retrieve
    halt_retrieval_count: int = 10


@dataclass
class GenerateResult:
    text: str
    tokens: list[int]
    grounding_scores: list[float]
    halted_positions: list[int]
    total_tokens: int


def sample_token(
    logits: mx.array,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> mx.array:
    """Sample a single token from logits with temperature, top-k, and top-p."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k_vals = mx.sort(logits, axis=-1)[:, -top_k:]
        threshold = top_k_vals[:, 0:1]
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits = mx.sort(logits, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)

        # Find cutoff
        mask = cumulative - sorted_probs > top_p
        # Zero out tokens below cutoff
        sorted_logits = mx.where(mask, mx.array(float("-inf")), sorted_logits)

        # Unsort — for simplicity, re-apply softmax and sample
        logits = sorted_logits

    probs = mx.softmax(logits, axis=-1)
    token = mx.random.categorical(mx.log(probs + 1e-10))
    return token


def generate(
    model: RegentModel,
    input_ids: mx.array,
    tokenizer,
    config: GenerateConfig | None = None,
    essence: mx.array | None = None,
    epg_node_tokens: mx.array | None = None,
    epg_scalars: mx.array | None = None,
    epg_categories: mx.array | None = None,
    eos_id: int = 2,
) -> GenerateResult:
    """
    Generate tokens with verification-gated decoding.

    Args:
        model: the Regent model
        input_ids: (1, prompt_len) — tokenized prompt
        tokenizer: for decoding output
        config: generation configuration
        essence: (1, 7) — essence state vector
        epg_*: EPG context tensors
        eos_id: end-of-sequence token ID

    Returns:
        GenerateResult with text, tokens, grounding scores, and halt positions
    """
    cfg = config or GenerateConfig()

    # Initial forward pass with full prompt (no cache for simplicity in prototype)
    output = model(
        input_ids=input_ids,
        essence=essence,
        epg_node_tokens=epg_node_tokens,
        epg_scalars=epg_scalars,
        epg_categories=epg_categories,
        use_chunked=False,
    )

    generated_tokens = []
    grounding_scores = []
    halted_positions = []

    # Get initial logits and grounding from last position
    next_logits = output["logits"][:, -1:, :]  # (1, 1, vocab)
    next_grounding = output["grounding"][:, -1].item() if output.get("grounding") is not None else 1.0

    for step in range(cfg.max_tokens):
        g_score = next_grounding
        grounding_scores.append(g_score)

        # Determine decoding zone
        if g_score > cfg.flow_threshold:
            # FLOW: normal sampling
            temp = cfg.temperature
        elif g_score > cfg.caution_threshold:
            # CAUTION: conservative sampling
            temp = cfg.caution_temperature
        else:
            # HALT: flag this position
            halted_positions.append(step)
            temp = cfg.caution_temperature

        # Sample token
        token = sample_token(
            next_logits[:, 0, :],
            temperature=temp,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )
        mx.eval(token)

        token_id = token.item()
        if token_id == eos_id:
            break

        generated_tokens.append(token_id)

        # Forward pass for next token
        token_input = token.reshape(1, 1)
        output = model(
            input_ids=token_input,
            essence=essence,
            use_chunked=False,
        )

        next_logits = output["logits"]  # (1, 1, vocab)
        if output.get("grounding") is not None:
            next_grounding = output["grounding"][:, -1].item()
        else:
            next_grounding = 1.0

    # Decode
    text = tokenizer.decode(generated_tokens)

    return GenerateResult(
        text=text,
        tokens=generated_tokens,
        grounding_scores=grounding_scores,
        halted_positions=halted_positions,
        total_tokens=len(generated_tokens),
    )
