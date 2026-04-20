"""HuggingFace Transformers integration for structured generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    import torch

from .guide import RegexGuide, STATE_DONE
from .json_schema import model_to_regex
from .regex_parser import DFA


class StructuredLogitsProcessor:
    """``LogitsProcessor`` that masks tokens not allowed by a :class:`RegexGuide`.

    Designed for ``batch_size == 1`` (greedy / sampling). Beam search is
    rejected by :func:`generate` because only the first beam would be
    structurally constrained.
    """

    def __init__(self, guide: RegexGuide, prompt_length: int):
        self.guide = guide
        self.prompt_length = prompt_length
        self.current_state: int = guide.start_state
        self._prev_len: int = prompt_length

    def __call__(
        self,
        input_ids: "torch.LongTensor",
        scores: "torch.FloatTensor",
    ) -> "torch.FloatTensor":
        import torch

        # input_ids: (batch, seq_len)   scores: (batch, vocab)
        seq_len = input_ids.shape[-1]

        # Advance DFA state for every token generated since last call.
        for i in range(self._prev_len, seq_len):
            tid = input_ids[0, i].item()
            self.current_state = self.guide.advance(self.current_state, tid)
        self._prev_len = seq_len

        # If we already finished (post-EOS in accept state), allow only EOS.
        if self.current_state == STATE_DONE:
            mask = torch.full_like(scores, float("-inf"))
            if self.guide.eos_token_id is not None:
                mask[:, self.guide.eos_token_id] = 0.0
            return scores + mask

        if self.current_state == DFA.DEAD:
            raise RuntimeError(
                "StructuredLogitsProcessor: the DFA entered the DEAD state — "
                "a generated token produced a character not accepted by the "
                "schema regex. The structural guarantee cannot be upheld."
            )

        # Build mask: only valid tokens keep their scores.
        valid = self.guide.get_valid_tokens(self.current_state)
        if not valid:
            raise RuntimeError(
                f"StructuredLogitsProcessor: no valid continuation tokens from "
                f"DFA state {self.current_state}. The structural guarantee "
                "cannot be upheld; check the schema and tokenizer coverage."
            )

        mask = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(valid, dtype=torch.long, device=scores.device)
        mask[:, idx] = 0.0
        return scores + mask


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def generate(
    model,
    tokenizer,
    schema: Type,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    verbose: bool = False,
    **generate_kwargs,
) -> str:
    """Generate text constrained to the JSON schema of *schema*.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A HuggingFace causal-LM.
    tokenizer : transformers.PreTrainedTokenizer
        Matching tokenizer.
    schema : type[BaseModel]
        Pydantic model whose JSON schema defines the structure.
    prompt : str
        Text prompt that precedes the generated JSON.
    max_new_tokens : int
        Maximum number of tokens to generate.
    verbose : bool
        Print progress during guide precomputation.
    **generate_kwargs
        Extra keyword arguments forwarded to ``model.generate()``.

    Returns
    -------
    str
        The generated JSON string (validated against *schema*).
    """
    if generate_kwargs.get("num_beams", 1) > 1:
        raise ValueError(
            "Beam search (num_beams > 1) is not supported by "
            "StructuredLogitsProcessor: only the first beam would be "
            "structurally constrained, so other beams could produce invalid "
            "output. Use greedy decoding or sampling instead."
        )

    pattern = model_to_regex(schema)
    if verbose:
        print(f"[structured_outlines] regex ({len(pattern)} chars): {pattern[:120]}…")

    guide = RegexGuide(pattern, tokenizer, verbose=verbose)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    processor = StructuredLogitsProcessor(guide, prompt_len)

    generate_kwargs.setdefault("do_sample", False)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=[processor],
        pad_token_id=tokenizer.eos_token_id,
        **generate_kwargs,
    )

    generated_ids = output[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)