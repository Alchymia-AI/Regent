"""
Architecture validation tests.

Validates that all components build, forward pass runs, gradients flow,
and the model produces expected output shapes.
"""

import sys
sys.path.insert(0, "/Users/morningstar/Documents/Others/Synthetic-Conciousness/regent-model")

import mlx.core as mx
import mlx.nn as nn

from regent_model.blocks.mamba2 import Mamba2Block, Mamba2Config
from regent_model.blocks.attention import GQABlock
from regent_model.heads.gen_head import GenHead
from regent_model.heads.ver_head import VerHead
from regent_model.encoder.epg_encoder import EPGEncoder
from regent_model.layers.model import RegentModel, RegentConfig


def test_mamba2_block():
    print("Testing Mamba2Block...", end=" ")
    cfg = Mamba2Config(d_model=64, d_state=16, d_conv=4, expand=2, n_heads=4)
    block = Mamba2Block(cfg)

    x = mx.random.normal((2, 32, 64))  # batch=2, seq=32, d_model=64

    # Sequential scan
    y_seq, _ = block(x, use_chunked=False)
    mx.eval(y_seq)
    assert y_seq.shape == (2, 32, 64), f"Expected (2, 32, 64), got {y_seq.shape}"

    # Chunked SSD
    y_chunk, _ = block(x, use_chunked=True)
    mx.eval(y_chunk)
    assert y_chunk.shape == (2, 32, 64), f"Expected (2, 32, 64), got {y_chunk.shape}"

    print("OK")


def test_gqa_block():
    print("Testing GQABlock...", end=" ")
    block = GQABlock(d_model=64, n_q_heads=8, n_kv_heads=2, head_dim=8, window_size=32)

    x = mx.random.normal((2, 16, 64))
    y, cache = block(x)
    mx.eval(y)
    assert y.shape == (2, 16, 64), f"Expected (2, 16, 64), got {y.shape}"

    # Test with cache (incremental decoding)
    x2 = mx.random.normal((2, 1, 64))
    y2, cache2 = block(x2, cache=cache)
    mx.eval(y2)
    assert y2.shape == (2, 1, 64), f"Expected (2, 1, 64), got {y2.shape}"
    print("OK")


def test_gen_head():
    print("Testing GenHead...", end=" ")
    head = GenHead(d_model=64, vocab_size=256)
    h = mx.random.normal((2, 16, 64))
    logits = head(h)
    mx.eval(logits)
    assert logits.shape == (2, 16, 256), f"Expected (2, 16, 256), got {logits.shape}"
    print("OK")


def test_ver_head():
    print("Testing VerHead...", end=" ")
    head = VerHead(d_model=64, hidden_dim=32)
    h = mx.random.normal((2, 16, 64))
    scores = head(h)
    mx.eval(scores)
    assert scores.shape == (2, 16), f"Expected (2, 16), got {scores.shape}"

    # Scores should be in [0, 1] (sigmoid output)
    assert mx.all(scores >= 0.0).item(), "Scores should be >= 0"
    assert mx.all(scores <= 1.0).item(), "Scores should be <= 1"
    print("OK")


def test_epg_encoder():
    print("Testing EPGEncoder...", end=" ")
    encoder = EPGEncoder(
        d_model=64, scalar_features=5, n_categories=15,
        category_embed_dim=8, n_encoder_layers=1, encoder_heads=2,
    )

    node_tokens = mx.random.normal((2, 4, 8, 64))  # batch=2, 4 nodes, 8 tokens each, pre-embedded
    scalars = mx.random.normal((2, 4, 5))
    categories = mx.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=mx.int32)

    prefix = encoder(node_tokens, scalars, categories)
    mx.eval(prefix)
    assert prefix.shape == (2, 4, 64), f"Expected (2, 4, 64), got {prefix.shape}"
    print("OK")


def test_full_model():
    print("Testing RegentModel (370M config)...", end=" ")
    # Use a tiny config for testing
    cfg = RegentConfig(
        d_model=64,
        n_layer=8,
        vocab_size=256,
        ssm_expand=2,
        ssm_d_state=16,
        ssm_d_conv=4,
        ssm_n_heads=4,
        ssm_chunk_size=16,
        attn_layers=(3, 7),  # 2 attention layers
        attn_n_q_heads=4,
        attn_n_kv_heads=2,
        attn_head_dim=16,
        attn_window_size=32,
        tie_embeddings=True,
        ver_enabled=True,
        ver_hidden_dim=32,
        epg_max_nodes=4,
        epg_scalar_features=5,
        epg_n_categories=15,
        epg_category_embed_dim=4,
        epg_n_encoder_layers=1,
        epg_encoder_heads=2,
        essence_input_dim=7,
        essence_inject_every_n=4,
    )

    model = RegentModel(cfg)
    params = model.count_parameters()
    print(f"({params['total_millions']}M params)", end=" ")

    # Forward pass without EPG
    input_ids = mx.zeros((2, 16), dtype=mx.int32)
    essence = mx.zeros((2, 7))

    output = model(input_ids=input_ids, essence=essence, use_chunked=False)
    mx.eval(output["logits"])

    assert output["logits"].shape == (2, 16, 256), f"Logits shape wrong: {output['logits'].shape}"
    assert output["grounding"].shape == (2, 16), f"Grounding shape wrong: {output['grounding'].shape}"
    print("OK")


def test_full_model_with_epg():
    print("Testing RegentModel with EPG prefix...", end=" ")
    cfg = RegentConfig(
        d_model=64, n_layer=4, vocab_size=256,
        ssm_expand=2, ssm_d_state=16, ssm_d_conv=4, ssm_n_heads=4,
        attn_layers=(3,), attn_n_q_heads=4, attn_n_kv_heads=2,
        attn_head_dim=16, attn_window_size=32,
        ver_enabled=True, ver_hidden_dim=32,
        epg_max_nodes=4, epg_scalar_features=5, epg_n_categories=15,
        epg_category_embed_dim=4, epg_n_encoder_layers=1, epg_encoder_heads=2,
        essence_input_dim=7, essence_inject_every_n=2,
    )
    model = RegentModel(cfg)

    input_ids = mx.zeros((2, 16), dtype=mx.int32)
    essence = mx.zeros((2, 7))
    epg_tokens = mx.zeros((2, 4, 8), dtype=mx.int32)  # 4 nodes, 8 tokens each
    epg_scalars = mx.random.normal((2, 4, 5))
    epg_categories = mx.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=mx.int32)

    output = model(
        input_ids=input_ids,
        essence=essence,
        epg_node_tokens=epg_tokens,
        epg_scalars=epg_scalars,
        epg_categories=epg_categories,
        use_chunked=False,
    )
    mx.eval(output["logits"])

    # Logits should be for the token sequence only (not EPG prefix)
    assert output["logits"].shape == (2, 16, 256), f"Logits shape: {output['logits'].shape}"
    assert output["grounding"].shape == (2, 16), f"Grounding shape: {output['grounding'].shape}"
    print("OK")


def test_gradient_flow():
    print("Testing gradient flow...", end=" ")
    cfg = RegentConfig(
        d_model=64, n_layer=4, vocab_size=256,
        ssm_expand=2, ssm_d_state=16, ssm_d_conv=4, ssm_n_heads=4,
        attn_layers=(3,), attn_n_q_heads=4, attn_n_kv_heads=2,
        attn_head_dim=16, attn_window_size=32,
        ver_enabled=True, ver_hidden_dim=32,
        epg_max_nodes=4, epg_scalar_features=5, epg_n_categories=15,
        epg_category_embed_dim=4, epg_n_encoder_layers=1, epg_encoder_heads=2,
        essence_input_dim=7, essence_inject_every_n=2,
    )
    model = RegentModel(cfg)

    input_ids = mx.zeros((2, 16), dtype=mx.int32)
    labels = mx.zeros((2, 16), dtype=mx.int32)

    def loss_fn(model):
        output = model(input_ids=input_ids, use_chunked=False)
        logits = output["logits"].reshape(-1, 256)
        targets = labels.reshape(-1)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss)

    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 10, f"Loss suspiciously high: {loss.item()}"
    print(f"OK (loss={loss.item():.3f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Regent Model Architecture Tests")
    print("=" * 60)

    test_mamba2_block()
    test_gqa_block()
    test_gen_head()
    test_ver_head()
    test_epg_encoder()
    test_full_model()
    test_full_model_with_epg()
    test_gradient_flow()

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
