from typing import List, Tuple
import argparse
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
import utils
import copy
import numpy as np
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools
from .icl import get_icl_prompts, do_sample, get_performance_metric
import tqdm
import random

DEVICE = os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu"

if DEVICE == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif DEVICE == "gpu" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Fine-tuning using device: ", DEVICE)


class LoRAConv1DWrapper(nn.Module):
    def __init__(self, conv1dmodule: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = conv1dmodule

        ###
        ### Set up your LoRA-augmented layer here.
        ### You should initialize your parameters so that the residual matrix AB^T is zero,
        ###     but be careful how you do this (i.e., make sure you eventually get
        ###     non-zero gradients to both matrices during fine-tuning)!
        ### Initialization hint: what do the gradients look like after 1 and 2 steps of fine-tuning
        ###     if you initialize both A and B to zero? What about if just one is zero?
        ###
        
        self.lora_A, self.lora_B = None, None
        
        ### START CODE HERE ###
        for param in self.base_module.parameters():
            param.requires_grad = False

        out_features, in_features = conv1dmodule.weight.shape

        self.lora_A = nn.Parameter(torch.randn(out_features, lora_rank))
        self.lora_B = nn.Parameter(torch.randn(in_features, lora_rank))

        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)
        ### END CODE HERE ###


    def forward(self, x):
        ###
        ### Perform the forward pass of your LoRA-augmented layer here.
        ### Note: you don't need to ever explicitly construct the matrix AB^T.
        ### Hint: matrix multiplication is associative.
        ###
        #############################
        ### START CODE HERE ###
        original_output = self.base_module(x)

        lora_update = self.lora_A @ self.lora_B.T

        updated_output = original_output + x @ lora_update

        return updated_output
        ### END CODE HERE ###


def parameters_to_fine_tune(model: nn.Module, mode: str) -> List:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)
    
    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """

    # Helper function to flatten a list of lists
    def flatten_list(lst):
        return [item for sublist in lst for item in sublist]

    param_list = []

    if mode == 'all':
        ### START CODE HERE ###
        return [param for param in model.parameters() if param.requires_grad]
        ### END CODE HERE ###
    elif mode == 'last':
        ### START CODE HERE ###
        for block in model.transformer.h[-2:]:
            param_list.append([param for param in block.parameters() if param.requires_grad])
        ### END CODE HERE ###
    elif mode == 'first':
        ### START CODE HERE ###
        for block in model.transformer.h[:2]:
            param_list.append([param for param in block.parameters() if param.requires_grad])
        ### END CODE HERE ###
    elif mode == 'middle':
        ### START CODE HERE ###
        num_layers = len(model.transformer.h)
        middle_index = num_layers // 2
        for block in model.transformer.h[middle_index - 1:middle_index + 1]:
            param_list.append([param for param in block.parameters() if param.requires_grad])
        ### END CODE HERE ###
    elif mode.startswith('lora'):
        ### START CODE HERE ###
        for layer in model.transformer.h:
            for submodule in [layer.mlp.c_fc, layer.mlp.c_proj, layer.attn.c_attn]:
                if isinstance(submodule, LoRAConv1DWrapper):
                    param_list.append(submodule.lora_A)
                    param_list.append(submodule.lora_B)
        return param_list
        ### END CODE HERE ###
    else:
        raise NotImplementedError()

    return flatten_list(param_list)


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the cross-entropy loss for either sequence classification or generation.

    For generation, you'll need to deal with the fact that different sequences witihn
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average loss is the *average loss over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor
        of *UNNORMALIZED* logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A zero-dim tensor representing the average cross-entropy loss over all batch 
        elements (and sequence timesteps, if applicable)
    """

    loss = None
    if logits.dim() == 2:
        ### START CODE HERE ###
        loss = F.cross_entropy(logits, targets)
        ### END CODE HERE ###
    elif logits.dim() == 3:
        ### START CODE HERE ###
        shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shifted_targets = targets[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(shifted_logits, shifted_targets, ignore_index=-100)
        ### END CODE HERE ###
    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')

    return loss

def get_acc(logits, targets):
    """
    Computes the exact match accuracy for either sequence classification or generation. i.e.,
      the fraction of predictions for which the most likely class/token equals the target.

    For generation, you'll need to deal with the fact that different sequences within
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average accuracy is the *average accuracy over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor of logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A *scalar* representing the average exact-match accuracy over all non-masked batch 
        elements (and sequence timesteps, if applicable)
    """

    if logits.dim() == 2:
        ### START CODE HERE ###
        _, predicted = torch.max(logits, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        return accuracy
        ### END CODE HERE ###
    elif logits.dim() == 3:
        ### START CODE HERE ###
        _, predicted = torch.max(logits, 2)
        predicted = predicted[:, :-1]
        targets = targets[:, 1:]
        mask = targets != -100
        correct = ((predicted == targets) & mask).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
        ### END CODE HERE ###
    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')


def ft_bert(model, tok, x, y, mode, debug, batch_size=8):
    model = copy.deepcopy(model)

    if mode.startswith('lora'):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=1e-4)
    all_x = tok(x, return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tok([x[i] for i in batch], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if debug:
            break

        if step % 10 == 0:
            with torch.inference_mode():
                total_acc = get_acc(model(**all_x).logits, all_y)
            pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')
            if total_acc > 0.75:
                break
    return model


def tokenize_gpt2_batch(tokenizer, x, y):
    """
    Implement the tokenization step for a batch of examples for GPT-2.

    Args:
        tokenizer: a GPT2Tokenizer that you can call and receive a dictionary of:
          - input_ids: a list (or tensor) of token ids
          - attention_mask: a list (or tensor) of 1s and 0s indicating which tokens
              are padding (if you requested padding and tensors from the tokenizer)
        x: a list of strings, each of which is the input for a single example
        y: a list of strings, each of which is a *target* for a single example
    
    Returns:
        A dictionary with the following keys:
            - input_ids: a tensor of shape [batch_size, sequence_length] 
                containing the token ids
            - attention_mask: a tensor of shape [batch_size, sequence_length] 
                containing 1s and 0s indicating which tokens are padding
            - labels: a tensor of shape [batch_size, sequence_length] containing
                the target token ids, with -100 for non-target tokens (i.e., the
                tokens in the input part of each example or padding tokens)
        where sequence_length is determined by the (x, y) pair whose tokenized
        length is the longest in the batch. The other sequences should be padded to
        this length (you can get the tokenizer to handle this padding!).

    Example:
        >>> x = ['Who is the singer for the band Queen?', 'What is the capital of France?']
        >>> y = ['Freddie Mercury', 'Paris']
        >>> tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        >>> tokenizer_dict = tokenizer([x_ + y_ for x_, y_ in zip(x, y)], return_tensors='pt', padding=True)
        >>> tokenizer_dict['input_ids']
        tensor([[ 8241,   318,   262, 14015,   329,   262,  4097,  7542,    30, 30847, 11979, 21673],
                [ 2061,   318,   262,  3139,   286,  4881,    30, 40313, 50256, 50256, 50256, 50256]])
        >>> tokenizer_dict['attention_mask']
        tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        >>> tokenizer(x)['input_ids']
        [[8241, 318, 262, 14015, 329, 262, 4097, 7542, 30],
         [2061, 318, 262, 3139, 286, 4881, 30]]
        >>> tokenizer(y)['input_ids']
        [[30847, 11979, 21673],
         [40313]]

        In this case, our labels should look like:
        [[-100, -100, -100, -100, -100, -100, -100, -100,   -100,  30847, 11979, 21673],
         [-100, -100, -100, -100, -100, -100, -100,  40313, -100, -100,  -100,  -100]]
        Note we've replaced padding tokens and the input prefix for each example
            with -100, leaving only the tokens in y.

        Other note: you can add new keys (such as 'labels') to the dictionary
            returned by the tokenizer without creating a new dictionary.
    """
    tokenized_sequences = None

    ### START CODE HERE ###
    tokenized_sequences = tokenizer([x_ + y_ for x_, y_ in zip(x, y)], return_tensors='pt', padding=True)

    labels = torch.full_like(tokenized_sequences['input_ids'], -100)

    for i, (input_seq, target_seq) in enumerate(zip(x, y)):
        input_id_len = len(tokenizer(input_seq)['input_ids'])
        target_ids = tokenizer(target_seq, add_special_tokens=False)['input_ids']
        labels[i, input_id_len:input_id_len+len(target_ids)] = torch.tensor(target_ids)

    tokenized_sequences['labels'] = labels

    ### END CODE HERE ###
    
    return tokenized_sequences.to(DEVICE)


def add_prefixes(x: List[str], y: List[str], dataset: str) -> Tuple[List[str], List[str]]:
    input_prefix = '' if utils.is_qa_dataset(dataset) else ''
    label_prefix = ' In the' if utils.is_qa_dataset(dataset) else ' TL;DR:'
    label_suffix = '.' if utils.is_qa_dataset(dataset) else ''

    x = [input_prefix + x_.replace('\n', ' ') + label_prefix for x_ in x]
    y = [' ' + y_.replace('\n', ' ') + label_suffix for y_ in y]

    return x, y


def ft_gpt2(model, tok, x, y, mode, dataset, batch_size=8, grad_accum=8):
    x, y = add_prefixes(x, y, dataset)

    model = copy.deepcopy(model)

    if mode.startswith('lora'):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))
            m.attn.c_attn = LoRAConv1DWrapper(m.attn.c_attn, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=2e-5)
    all_both = tokenize_gpt2_batch(tok, x, y)
    max_n = len(x) * 10
    pbar = tqdm.tqdm(range(max_n))
    idxs = []
    for step in pbar:
        model.train()

        if len(idxs) < batch_size // grad_accum:
            idxs = list(range(len(x)))
            random.shuffle(idxs)
        batch_idxs = idxs[:batch_size // grad_accum]
        idxs = idxs[batch_size // grad_accum:]

        # Outline:
        # 1. Sample a random minibatch of examples of size batch_size // grad_accum using the batch_idxs variable
        # 2. Tokenize the batch using the tokenize_gpt2_batch function you implemented
        # 3. Run the model on the batch, get the logits, and compute the loss using the get_loss function you implemented
        #      *NOTE 1* Pass `use_cache=False` when you call model() to avoid a huggingface warning
        #      *NOTE 2* You MUST compute the loss using your get_loss function applied to the model_output.logits.
        #        Don't use the loss attribute of the model output for training (you will not get credit for this).
        #        However, you can use the loss attribute of the model output to test your get_loss function (they should match).
        # 4. Backpropagate the loss (divided by the grad_accum parameter)
        # 5. Take a step of the optimizer and zero the model gradients ***only every grad_accum steps***
        #    Be careful that you don't take a step after the very first backward pass (i.e., when step == 0)
        # Note: the ** operator will unpack a dictionary into keyword arguments to a function (such as your model)
        #############################
        ### START CODE HERE ###
        batch_x = [x[i] for i in batch_idxs]
        batch_y = [y[i] for i in batch_idxs]

        tokenized_batch = tokenize_gpt2_batch(tok, batch_x, batch_y)

        # Forward pass
        outputs = model(**tokenized_batch, use_cache=False)
        loss = get_loss(outputs.logits, tokenized_batch['labels'])

        # Backward pass
        adjusted_loss = loss / grad_accum

        # Proceed with backpropagation using adjusted_loss
        adjusted_loss.backward()

        if (step + 1) % grad_accum == 0:
            # Take optimizer step after accumulating gradients
            optimizer.step()
            optimizer.zero_grad()
        ### END CODE HERE ###

        if step % (grad_accum * 5) == 0:
            with torch.inference_mode():
                model.eval()
                accs = []
                for idx in range(len(list(all_both.values())[0])):
                    d = {k: v[idx:idx+1] for k, v in all_both.items()}
                    acc = get_acc(model(**d).logits, d['labels'])
                    accs.append(acc)
                total_acc = sum(accs) / len(accs)
                pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')

            if total_acc >= utils.early_stop_thresold(dataset):
                print('Early stopping!')
                break
    return model


def eval(model, tok, val_data):
    x = tok(val_data['x'], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    y = torch.tensor(val_data['y'], device=DEVICE)
    with torch.inference_mode():
        logits = model(**x).logits
    return get_acc(logits, y)


def run_ft(models: List[str], datasets: List[str], ks: List[int], modes: List[str], debug: bool, repeats: int, n_val: int = 125):
    results = {}
    for dataset in datasets:

        utils.fix_random_seeds()

        if debug:
            n_val = 1   
        train, val = utils.get_dataset(dataset, max(ks), n_val=n_val)
        for model_name, mode in itertools.product(models, modes):

            utils.fix_random_seeds()
            
            if dataset == 'amazon':
                model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForSequenceClassification, num_labels=5)
            else:
                model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)

            stop_tokens = utils.stop_tokens(tokenizer)


            for k in ks:
                print(f'Fine-tuning {model_name} on {dataset} with k={k} and mode={mode}')

                utils.fix_random_seeds()

                for repeat in range(repeats):
                    if repeat > 0:
                        print(f'Beginning repeat #{repeat}')
                    if dataset == 'amazon':
                        fine_tuned = ft_bert(model, tokenizer, train['x'][:k*5], train['y'][:k*5], mode, debug)
                        val_acc = eval(fine_tuned, tokenizer, val)
                        results['_'.join([model_name, dataset, str(k), mode])] = val_acc
                    else:
                        if k > 0:
                            fine_tuned = ft_gpt2(model, tokenizer, train['x'][:k], train['simple_y'][:k], mode, dataset)
                        else:
                            fine_tuned = copy.deepcopy(model)
                            fine_tuned.to(DEVICE)

                        fine_tuned.eval()
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val['x'])))))

                        for row in pbar:
                            test_input = val['x'][row]
                            targets.append(val['y'][row])
                            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
                            prompt_mode = 'qa' if utils.is_qa_dataset(dataset) else 'tldr'
                            prompt = get_icl_prompts([], [], test_input, prompt_mode=prompt_mode)
                            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
                            sampled_tokens = do_sample(fine_tuned, input_ids, stop_tokens, max_tokens)
                            decoded = tokenizer.decode(sampled_tokens).strip()
                            predictions.append(decoded)
                            metric = get_performance_metric(predictions, targets, utils.metric_for_dataset(dataset))
                            pbar.set_description(f'Eval: {metric:.04f}')
                        results['_'.join([model_name, dataset, str(k), mode])] = metric

                    print(results)
                    question = 'ft'
                    if not os.path.exists(f'submission/results/{question}'):
                        os.makedirs(f'submission/results/{question}')

                    for k_, v in results.items():
                        with open(f'submission/results/{question}/{k_}.json', 'w') as f:
                            json.dump({'metric': v}, f)
                    results = {}


def plot_ft(models, datasets, ks, modes, output):
    data = defaultdict(lambda: defaultdict(list))
    question = 'ft'

    x_vals = set()
    for dataset in datasets:
        for model, mode in itertools.product(models, modes):
            for k in ks:
                fn = '_'.join([model, dataset, str(k), mode])
                id_ = '_'.join([model, dataset, mode])
                with open(f'submission/results/{question}/{fn}.json', 'r') as f:
                    score = json.load(f)['metric']
                    data[id_]['x'].append(k)
                    x_vals.add(k)
                    data[id_]['y'].append(score)

        for k, v in data.items():
            plt.plot(v['x'], v['y'], label=k)

    if max(x_vals) > 4:
        plt.xscale('symlog')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(sorted(x_vals))
    plt.legend()
    plt.title(' & '.join(datasets))
    plt.ylabel('/'.join([utils.metric_for_dataset(dataset) for dataset in datasets]))
    plt.xlabel('Number of support examples')
    plt.savefig(output, bbox_inches='tight')
