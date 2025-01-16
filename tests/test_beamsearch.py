import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from multi_choices_parser import MultiChoicesParser
from divergent_beamsearch.algorithm import divergent_beamsearch, divergent_logprob, log1mexp
from multi_choices_parser import MultiChoicesParser, DEFAULT_END_SYMB


TEST_END_SYMBS = [DEFAULT_END_SYMB, 'tokenizer']

@pytest.fixture
def model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

@pytest.fixture
def fakemodel_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Define a small GPT-2 configuration
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,  # Use the default GPT-2 tokenizer vocab size
        n_positions=64,  # Maximum sequence length
        n_ctx=64,  # Context window size
        n_embd=8,  # Size of the embeddings
        n_layer=1,  # Number of layers
        n_head=2,  # Number of attention heads
    )

    # Instantiate a model with the custom configuration
    model = GPT2LMHeadModel(config)
    model.eval()

    return model, tokenizer

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("end_symb", TEST_END_SYMBS)
def test_divergent_beamsearch(model_and_tokenizer, device, end_symb):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")
    model, tokenizer = model_and_tokenizer
    model.to(device)
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    beam_size = 10
    max_length = 10
    pad_token_id = tokenizer.eos_token_id

    possible_answers = [' Paris', ' Madrid', ' Paris Hilton', ' Bri bra brouuu Mario Brooos']
    tokenized_answers = tokenizer(possible_answers).input_ids

    if end_symb == 'tokenizer':
        end_symb = tokenizer.eos_token_id
    
    multi_choices_parser = MultiChoicesParser([tokenized_answers], end_symb=end_symb)

    with torch.no_grad():
        logprob_paris = model(input_ids).logits.cpu().log_softmax(dim=-1)[0, -1, tokenized_answers[0][0]]
        logprob_hilton = model(torch.cat([input_ids, torch.tensor(tokenized_answers[2][0], device=device).view(1,1)], dim=-1)).logits.cpu().log_softmax(dim=-1)[0, -1, tokenized_answers[2][1]]
        logprob_paris_hilton = logprob_paris + logprob_hilton
        logprob_madrid = model(input_ids).logits.cpu().log_softmax(dim=-1)[0, -1, tokenized_answers[1][0]]
        logprob_paris_diverge = logprob_paris + log1mexp(logprob_hilton)
        input_garbage = torch.tensor(input_ids.tolist()[0] + tokenized_answers[-1]).unsqueeze(0).to(device)
        logsoftmax_garbage = model(input_garbage).logits.log_softmax(-1)
        logprob_garbage = torch.gather(logsoftmax_garbage[:, 4:-1, :], 2, input_garbage[:, 5:, None]).squeeze(-1).sum(-1)

    scores, solutions = divergent_beamsearch(
        input_ids=input_ids,
        model=model,
        beam_size=beam_size,
        max_length=max_length,
        parser=multi_choices_parser,
        pad_token_id=pad_token_id,
        num_solutions=beam_size,
        end_symb=end_symb
    )
    true_solutions = torch.nn.utils.rnn.pad_sequence([torch.tensor(ans) for ans in tokenized_answers], batch_first=True, padding_value=pad_token_id)
    assert (solutions == true_solutions).all(), "Beam search did not return the expected solutions"
    assert torch.isclose(scores[0], logprob_paris_diverge), "Beam search did not return the expected score"
    assert torch.isclose(scores[1], logprob_madrid), "Beam search did not return the expected score"
    assert torch.isclose(scores[2], logprob_paris_hilton), "Beam search did not return the expected score"
    assert torch.isclose(scores[3], logprob_garbage), "Beam search did not return the expected score"


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("end_symb", TEST_END_SYMBS)
def test_divergent_logprob(fakemodel_and_tokenizer, device, end_symb):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")
    model, tokenizer = fakemodel_and_tokenizer
    model.to(device)
    prompts = [
        "The capital of France is Paris",
        "The top model Paris Hilton"
    ]
    tokenizer.pad_token = tokenizer.eos_token
    inp = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inp.input_ids.to(device)
    attention_mask = inp.attention_mask.to(device)

    possible_answers = [' Paris', ' Paris Hilton']
    tokenized_answers = tokenizer(possible_answers).input_ids

    if end_symb == 'tokenizer':
        end_symb = tokenizer.eos_token_id

    multi_choices_parser = MultiChoicesParser([tokenized_answers], end_symb=end_symb)

    input_len = attention_mask.sum(-1).cpu()
    probs = divergent_logprob(input_ids, attention_mask, model, multi_choices_parser, start=input_len - torch.tensor([1,2]), end_symb=end_symb)
    
    input_ids_1st = tokenizer("The capital of France is Paris Hilton", return_tensors='pt').input_ids.to(device)
    logprobs_1st = model(input_ids_1st).logits.cpu().log_softmax(dim=-1)
    logprob_paris = logprobs_1st[0, input_ids_1st.shape[1]-3, tokenized_answers[1][0]] # P(Paris | The capital of France is)
    logprob_hilton = logprobs_1st[0, input_ids_1st.shape[1]-2, tokenized_answers[1][1]] # P(Hilton | The capital of France is Paris)

    input_ids_2nd = tokenizer("The top model Paris Hilton", return_tensors='pt').input_ids.to(device)
    logprobs_2nd = model(input_ids_2nd).logits.cpu().log_softmax(dim=-1)
    logprob_paris_hilton = logprobs_2nd[0, -3, tokenized_answers[1][0]] + logprobs_2nd[0, -2, tokenized_answers[1][1]] # P(Paris Hilton | The top model)

    assert torch.isclose(probs[0], logprob_paris + log1mexp(logprob_hilton)), "P_divergent(Paris | The capital of France is) is incorrect"
    assert torch.isclose(probs[1], logprob_paris_hilton), "P_divergent(Paris Hilton | The top model) is incorrect"

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
