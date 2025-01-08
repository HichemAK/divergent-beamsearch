import math
import torch
from transformers import GPT2LMHeadModel
from multi_choices_parser import MultiChoicesParser, end_symb


class Parser:
    def step(self, token):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError
    
    def copy(self):
        raise NotImplementedError

def get_parsers_tokens(parsers : list[Parser]) -> tuple[list, list[int]]:
    parsers_tokens = []
    can_end = []
    for parser in parsers:
        tokens = list(parser.next())
        if end_symb in tokens:
            can_end.append(True)
            tokens.remove(end_symb)
        else:
            can_end.append(False)
        parsers_tokens.append(tokens)
    return parsers_tokens, can_end

def apply_mask_tokens(pred : torch.Tensor, parsers_tokens):
    mask = torch.ones_like(pred, dtype=torch.bool)
    for tokens in parsers_tokens:
        mask[:, tokens] = False
    pred[mask] = -float('inf')
    return pred[~pred.isinf().all(dim=-1)]


def batched_inference_logits(model : GPT2LMHeadModel, input_ids : torch.Tensor, attention_mask : torch.Tensor | None = None, batch_size : int = 32) -> torch.Tensor:
    logits = []
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    for i in range(0, input_ids.shape[0], batch_size):
        logits.append(model(input_ids[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size]).logits)
    return torch.cat(logits, dim=0)

def select_mask(source : list, mask : list[bool]) -> list:
    assert len(source) == len(mask)
    return [x for x, m in zip(source, mask) if m]


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )




class AcceptEverythingParser(Parser):
    def __init__(self, vocab_size : int):
        self.vocab_size = vocab_size
        self.tokens = tuple(range(vocab_size))

    def step(self, token):
        pass

    def next(self):
        return self.tokens
    
    def copy(self):
        return self

@torch.no_grad()
def divergent_beamsearch(input_ids : torch.Tensor, model : GPT2LMHeadModel, beam_size : int, max_length : int, parser : Parser, pad_token_id : int, batch_size=32, num_solutions = None) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_ids.shape[0] == 1, "Batch size must be 1"
    device = input_ids.device
    input_ids = input_ids.cpu()
    
    if num_solutions is None:
        num_solutions = beam_size
    vanilla = parser is None
    if vanilla:
        parser = AcceptEverythingParser(model.config.vocab_size)

    parsers_unfinished = [parser]
    scores_finished = torch.tensor([], dtype=torch.float)
    solutions_finished = torch.tensor([], dtype=torch.long).view(0,0)
    
    input_ids_unfinished = input_ids
    scores_unfinished = torch.tensor([0.0], dtype=torch.float)
    solutions_unfinished = torch.tensor([], dtype=torch.long).view(1,0)

    
    for _ in range(max_length):
        if len(input_ids_unfinished) == 0:
            break
        pred = batched_inference_logits(model, input_ids_unfinished.to(device), batch_size=batch_size)[:, -1].cpu()
        parsers_tokens, can_end = get_parsers_tokens(parsers_unfinished)
        logprobs = torch.log_softmax(pred, dim=-1)
        logprobs_filtered = apply_mask_tokens(logprobs, parsers_tokens)
        if len(logprobs_filtered):
            topk = torch.topk(logprobs_filtered, beam_size, dim=-1) # shape (batch_size, beam_size)
            values = topk.values + scores_unfinished.unsqueeze(-1)
            topk_global = values.flatten().topk(beam_size)
            best_tokens_row = topk_global.indices // beam_size
            best_tokens, best_tokens_logprobs = topk.indices[best_tokens_row, topk_global.indices % beam_size], topk.values[best_tokens_row, topk_global.indices % beam_size]
            notinf = ~best_tokens_logprobs.isinf()
            best_tokens, best_tokens_row, best_tokens_logprobs = best_tokens[notinf], best_tokens_row[notinf], best_tokens_logprobs[notinf]
        else:
            best_tokens = torch.tensor([], dtype=torch.long)
            best_tokens_row = torch.tensor([], dtype=torch.long)
            best_tokens_logprobs = torch.tensor([], dtype=torch.float)


        scores_finished_current = scores_unfinished[can_end]
        solutions_finished_current = solutions_unfinished[can_end]
        scores_finished_current = scores_finished_current + log1mexp(logprobs[can_end, select_mask(parsers_tokens, can_end)].logsumexp(dim=-1)).squeeze(-1)
        scores_finished = torch.cat([scores_finished, scores_finished_current])
        if len(solutions_finished_current):
            pad = torch.full((len(scores_finished_current), solutions_finished_current.shape[1] - solutions_finished.shape[1]), pad_token_id, dtype=torch.long)
            solutions_finished = torch.cat([solutions_finished.view(-1, solutions_finished_current.shape[1]+pad.shape[1]), torch.cat([solutions_finished_current, pad], dim=1)], dim=0)
        if solutions_finished.numel():
            # Keep num_solutions best solutions in finished
            order = scores_finished.argsort(descending=True)
            solutions_finished = solutions_finished[order][:num_solutions]
            scores_finished = scores_finished[order][:num_solutions]


        input_ids_unfinished = torch.cat([input_ids_unfinished[best_tokens_row], best_tokens.unsqueeze(-1)], dim=-1)
        scores_unfinished = scores_unfinished[best_tokens_row] + best_tokens_logprobs
        solutions_unfinished = torch.cat([solutions_unfinished[best_tokens_row], best_tokens.unsqueeze(-1)], dim=-1)
        parsers_unfinished = [parsers_unfinished[row].copy() for row in best_tokens_row]
        for parser, token in zip(parsers_unfinished, best_tokens.tolist()):
            parser.step(token)

    # Special case of vanilla beam search where all answers are valid
    if vanilla:
        order = scores_unfinished.argsort(descending=True)
        scores_finished = scores_unfinished[order][:num_solutions]
        solutions_finished = solutions_unfinished[order][:num_solutions]
    
    return scores_finished, solutions_finished


def set_slice_row(x : torch.Tensor, slices : torch.IntTensor, value) -> torch.Tensor:
    indices = [torch.arange(start, end) for start, end in slices]
    for i in range(slices.size(0)):
        x[i].index_fill_(0, indices[i], 0)

@torch.no_grad()
def divergent_logprob(input_ids : torch.Tensor, attention_mask : torch.Tensor | None, model : GPT2LMHeadModel, parsers : Parser | list[Parser] | None, batch_size=32, start : int | torch.IntTensor = None) -> torch.FloatTensor:
    if start is None:
        start = 0
    if isinstance(start, int):
        start = torch.tensor([start]*input_ids.shape[0])
    assert start.shape[0] == input_ids.shape[0]
    # -1 because next token offset
    start = start - 1

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    logits = batched_inference_logits(model, input_ids, attention_mask, batch_size).cpu()
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()

    logsoftmax = torch.log_softmax(logits, dim=-1)
    log_probs = torch.gather(
        logsoftmax[:, :-1, :], 2, input_ids[:, 1:, None]
    ).squeeze(-1)
    mask = attention_mask[:, 1:].cpu().clone()

    input_len = attention_mask.sum(-1)
    pos = torch.stack([torch.zeros_like(start), start], dim=-1)
    pos_anti = pos.flip(1)
    pos_anti[:, -1] = input_len
    set_slice_row(mask, pos, 0)
    vanilla_prob = (log_probs * mask).sum(-1)
    if parsers is None:
        parsers = AcceptEverythingParser(model.config.vocab_size)
    if not isinstance(parsers, (tuple, list)):
        parsers = [parsers.copy() for _ in range(len(input_ids))]
    next_possible_tokens = []
    for i, parser in enumerate(parsers):
        # +1 because no next-token offset
        start = pos_anti[i,0]+1
        for input_id, att in zip(input_ids[i, start:].tolist(), attention_mask[i, start:].tolist()):
            if not att:
                break
            parser.step(input_id)
        next_tokens = list(parser.next())
        try:
            next_tokens.remove(end_symb)
        except ValueError:
            pass
        next_possible_tokens.append(next_tokens)
    last_token_log_probs = torch.stack([log1mexp(logsoftmax[i, input_len[i]-1, tokens].logsumexp(-1)).squeeze() for i, tokens in enumerate(next_possible_tokens)])
    prob = vanilla_prob + last_token_log_probs
    return prob
