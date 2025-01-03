import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from multi_choices_parser import MultiChoicesParser
from divergent_beamsearch.algorithm import divergent_beamsearch, log1mexp

@pytest.fixture
def model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def test_divergent_beamsearch(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    beam_size = 5
    max_length = 10
    pad_token_id = tokenizer.eos_token_id

    possible_answers = [' Paris', ' Paris Hilton']
    tokenized_answers = tokenizer(possible_answers).input_ids
    multi_choices_parser = MultiChoicesParser([tokenized_answers])

    logprob_paris = model(input_ids).logits.log_softmax(dim=-1)[0, -1, tokenized_answers[0][0]]
    logprob_hilton = model(torch.cat([input_ids, torch.tensor(tokenized_answers[1][0]).view(1,1)], dim=-1)).logits.log_softmax(dim=-1)[0, -1, tokenized_answers[1][1]]
    logprob_paris_hilton = logprob_paris + logprob_hilton

    scores, solutions = divergent_beamsearch(
        input_ids=input_ids,
        model=model,
        beam_size=beam_size,
        max_length=max_length,
        multi_choices_parser=multi_choices_parser,
        pad_token_id=pad_token_id,
        num_solutions=10
    )
    true_solutions = torch.nn.utils.rnn.pad_sequence([torch.tensor(ans) for ans in tokenized_answers], batch_first=True, padding_value=pad_token_id)
    assert (solutions == true_solutions).all(), "Beam search did not return the expected solutions"
    assert scores[0] == logprob_paris + log1mexp(logprob_hilton), "Beam search did not return the expected score"
    assert scores[1] == logprob_paris_hilton, "Beam search did not return the expected score"