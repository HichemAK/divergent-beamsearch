import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from multi_choices_parser import MultiChoicesParser
from divergent_beamsearch.algorithm import divergent_beamsearch, log1mexp
from multi_choices_parser import MultiChoicesParser

@pytest.fixture
def model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_divergent_beamsearch(model_and_tokenizer, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")
    model, tokenizer = model_and_tokenizer
    model.to(device)
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    beam_size = 5
    max_length = 10
    pad_token_id = tokenizer.eos_token_id

    possible_answers = [' Paris', ' Paris Hilton']
    tokenized_answers = tokenizer(possible_answers).input_ids
    multi_choices_parser = MultiChoicesParser([tokenized_answers])

    logprob_paris = model(input_ids).logits.cpu().log_softmax(dim=-1)[0, -1, tokenized_answers[0][0]]
    logprob_hilton = model(torch.cat([input_ids, torch.tensor(tokenized_answers[1][0], device=device).view(1,1)], dim=-1)).logits.cpu().log_softmax(dim=-1)[0, -1, tokenized_answers[1][1]]
    logprob_paris_hilton = logprob_paris + logprob_hilton

    scores, solutions = divergent_beamsearch(
        input_ids=input_ids,
        model=model,
        beam_size=beam_size,
        max_length=max_length,
        parser=multi_choices_parser,
        pad_token_id=pad_token_id,
        num_solutions=10
    )
    true_solutions = torch.nn.utils.rnn.pad_sequence([torch.tensor(ans) for ans in tokenized_answers], batch_first=True, padding_value=pad_token_id)
    assert (solutions == true_solutions).all(), "Beam search did not return the expected solutions"
    assert scores[0] == logprob_paris + log1mexp(logprob_hilton), "Beam search did not return the expected score"
    assert scores[1] == logprob_paris_hilton, "Beam search did not return the expected score"

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_vanilla_beamsearch(model_and_tokenizer, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")
    # Verify that divergent beam search where all answers are valid is equivalent to vanilla beam search 
    # Results of beam search were compared with huggingface implementation (https://huggingface.co/spaces/m-ric/beam_search_visualizer)
    model, tok = model_and_tokenizer
    model.to(device)
    model.eval()
    prompt = "The capital of France is"
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    scores, sequences = divergent_beamsearch(
        input_ids, model, beam_size=3, max_length=1, pad_token_id=tok.eos_token_id, num_solutions=3, parser=None
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [" the", " now", " a"]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-2.4699, -3.0377, -3.0756]), atol=0.0001
    ).all()

    scores, sequences = divergent_beamsearch(
        input_ids, model, beam_size=3, max_length=2, pad_token_id=tok.eos_token_id, num_solutions=3, parser=None
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [" the capital", " now home", " now the"]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-4.2437, -5.3013, -5.3408]), atol=0.0001
    ).all()

    scores, sequences = divergent_beamsearch(
        input_ids, model, beam_size=3, max_length=3, pad_token_id=tok.eos_token_id, num_solutions=3, parser=None
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [" the capital of", " now home to", " now the capital"]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-4.3194, -5.3057, -7.7173]), atol=0.0001
    ).all()

    scores, sequences = divergent_beamsearch(
        input_ids, model, beam_size=3, max_length=4, pad_token_id=tok.eos_token_id, num_solutions=3, parser=None
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [
        " the capital of the",
        " the capital of France",
        " the capital of a",
    ]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-5.5825, -5.9150, -7.1716]), atol=0.0001
    ).all()

    scores, sequences = divergent_beamsearch(
        input_ids, model, beam_size=3, max_length=5, pad_token_id=tok.eos_token_id, num_solutions=3, parser=None
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [
        " the capital of France,",
        " the capital of France.",
        " the capital of the French",
    ]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-6.9453, -7.1549, -7.5727]), atol=0.0001
    ).all()


    scores, sequences = divergent_beamsearch(
        input_ids, model, beam_size=3, max_length=6, pad_token_id=tok.eos_token_id, num_solutions=3, parser=None
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [
        " the capital of France, and",
        " the capital of the French Republic",
        " the capital of France. It",
    ]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-8.1361, -8.7745, -9.1053]), atol=0.0001
    ).all()