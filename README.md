# Divergent Beam Search
## Overview

Divergent Beam Search is a variant of the beam search algorithm. Unlike the beam search where answers are constrained, which aims to find the answers with the highest probability of appearing, Divergent Beam Search focuses on finding answers that are not likely to be continued with another answer. Essentially, it finds the answers that maximize the probability of generating an answer before diverging into another subject given the prompt.

The core idea of this algorithm can be roughly summarized in the following optimization problem:

$$\max_{ans \in A} P(ans + diverging\ into\ another\ subject \mid prompt)$$

It is important that the set of answers $A$ is sufficiently exhaustive for this method to work. Otherwise, the algorithm could unjustifiably conclude that an answer is not being followed by the answer while this longer answer exists but is not included in the set $A$.   

## Installation

To install the package, use the following command:

```bash
pip install divergent-beamsearch
```

## Usage

Here's a brief example of how to use `divergent-beamsearch`:

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from multi_choices_parser import MultiChoicesParser
from divergent_beamsearch import divergent_beamsearch

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define input prompt
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Define beam search parameters
beam_size = 5
max_length = 10
pad_token_id = tokenizer.eos_token_id

# Define possible answers
possible_answers = [' Paris', ' Paris Hilton']
tokenized_answers = tokenizer(possible_answers).input_ids
multi_choices_parser = MultiChoicesParser([tokenized_answers])

# Perform beam search
scores, solutions = divergent_beamsearch(
    input_ids=input_ids,
    model=model,
    beam_size=beam_size,
    max_length=max_length,
    multi_choices_parser=multi_choices_parser,
    pad_token_id=pad_token_id,
    num_solutions=2
)

# Decode solutions
decoded_solutions = [tokenizer.decode(solution, skip_special_tokens=True) for solution in solutions]
print("Scores:", scores)
print("Solutions:", decoded_solutions)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
