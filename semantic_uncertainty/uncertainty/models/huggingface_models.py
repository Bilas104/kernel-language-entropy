"""Implement HuggingfaceModel models."""
import copy
import logging
from collections import Counter
import torch

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        # if 'llama' in model_name.lower():
        #     if model_name.endswith('-8bit'):
        #         kwargs = {'quantization_config': BitsAndBytesConfig(
        #             load_in_8bit=True,)}
        #         model_name = model_name[:-len('-8bit')]
        #         eightbit = True
        #     else:
        #         kwargs = {}
        #         eightbit = False

        #     if 'Llama-2' in model_name or "Llama-3" in model_name:
        #         base = 'meta-llama'
        #         if 'Llama-2' in model_name:
        #             model_name = model_name + '-hf'
        #     else:
        #         base = 'huggyllama'

        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         f"{base}/{model_name}", device_map="auto",
        #         token_type_ids=None)

        #     llama65b = '65b' in model_name and base == 'huggyllama'
        #     llama2_70b = '70b' in model_name and base == 'meta-llama'

        #     if ('7b' in model_name or "8B" in model_name or '13b' in model_name) or eightbit:
        #         self.model = AutoModelForCausalLM.from_pretrained(
        #             f"{base}/{model_name}", device_map="auto",
        #             max_memory={0: '80GIB'}, **kwargs,)

        #     elif llama2_70b or llama65b:
        #         path = snapshot_download(
        #             repo_id=f'{base}/{model_name}',
        #             allow_patterns=['*.json', '*.model', '*.safetensors'],
        #             ignore_patterns=['pytorch_model.bin.index.json']
        #         )

        #         config = AutoConfig.from_pretrained(f"{base}/{model_name}")
        #         with accelerate.init_empty_weights():
        #             self.model = AutoModelForCausalLM.from_config(config)
        #         self.model.tie_weights()
        #         max_mem = 15 * 4686198491

        #         device_map = accelerate.infer_auto_device_map(
        #             self.model.model,
        #             max_memory={0: max_mem, 1: max_mem},
        #             dtype='float16'
        #         )
        #         device_map = remove_split_layer(device_map)
        #         full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
        #         full_model_device_map["lm_head"] = 0

        #         self.model = accelerate.load_checkpoint_and_dispatch(
        #             self.model, path, device_map=full_model_device_map,
        #             dtype='float16', skip_keys='past_key_values')
        #     else:
        #         raise ValueError

        # elif 'mistral' in model_name.lower():

        #     if model_name.endswith('-8bit'):
        #         kwargs = {'quantization_config': BitsAndBytesConfig(
        #             load_in_8bit=True,)}
        #         model_name = model_name[:-len('-8bit')]
        #     if model_name.endswith('-4bit'):
        #         kwargs = {'quantization_config': BitsAndBytesConfig(
        #             load_in_4bit=True,)}
        #         model_name = model_name[:-len('-4bit')]
        #     else:
        #         kwargs = {}

        #     model_id = f'mistralai/{model_name}'
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         model_id, device_map='auto', token_type_ids=None,
        #         clean_up_tokenization_spaces=False)

        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         model_id,
        #         device_map='auto',
        #         max_memory={0: '80GIB'},
        #         **kwargs,
        #     )

        # elif 'falcon' in model_name:
        #     model_id = f'tiiuae/{model_name}'
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         model_id, device_map='auto', token_type_ids=None,
        #         clean_up_tokenization_spaces=False)

        #     kwargs = {'quantization_config': BitsAndBytesConfig(
        #         load_in_8bit=True,)}

        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         model_id,
        #         trust_remote_code=True,
        #         device_map='auto',
        #         **kwargs,
        #     )
        # else:
        #     raise ValueError

        # modification to better handle quantization suffixes and model ids.
        kwargs = {}
        quantization_suffix = None
        base_model_name = model_name

        # Unified logic to handle quantization suffixes correctly
        if base_model_name.endswith('-int8') or base_model_name.endswith('-8bit'):
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            quantization_suffix = '-int8' if base_model_name.endswith('-int8') else '-8bit'
            logging.info(f"Preparing to load model in 8-bit mode.")
        elif base_model_name.endswith('-4bit'):
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
            quantization_suffix = '-4bit'
            logging.info(f"Preparing to load model in 4-bit mode.")

        # If a suffix was found, remove it from the model name to find the base model
        if quantization_suffix:
            base_model_name = base_model_name[:-len(quantization_suffix)]

        # Determine the full model ID on Hugging Face Hub
        if 'llama' in base_model_name.lower():
            base = 'meta-llama' if ('Llama-2' in base_model_name or "Llama-3" in base_model_name) else 'huggyllama'
            if 'Llama-2' in base_model_name:
                base_model_name = base_model_name + '-hf'
            model_id = f"{base}/{base_model_name}"
        elif 'mistral' in base_model_name.lower():
            model_id = f"mistralai/{base_model_name}"
        elif 'falcon' in base_model_name.lower():
            model_id = f"tiiuae/{base_model_name}"
        else:
            # Fallback for other models like T5
            model_id = base_model_name
        
        logging.info(f"Loading model from Hugging Face ID: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the model with the prepared arguments
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True if 'falcon' in base_model_name.lower() else False,
            **kwargs
        )

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = 4096 if 'Llama-2' in model_name or "Llama-3" in model_name else 2048

    def predict(self, input_data, temperature, return_full=False):

        # Implement prediction.
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            if 'token_type_ids' in inputs:  # Some HF models have changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            #raise ValueError('Have not tested this in a while.')
            logging.error(f"Full answer should start from input_data. Setting input_data offset to 0")
            logging.error(f"Full answer is {full_answer}")
            logging.error(f"Input data is {input_data}")
            input_data_offset = 0

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                if 'falcon' not in self.model_name.lower():
                    raise ValueError(error_msg)
                else:
                    logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()

        # Get the number of tokens until the stop word comes up.
        # Note: Indexing with `stop_at` already excludes the stop_token.
        # Note: It's important we do this with full answer, since there might be
        # non-trivial interactions between the input_data and generated part
        # in tokenization (particularly around whitespaces.)
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        # Get the last hidden state (last layer) and the last token's embedding of the answer.
        # Note: We do not want this to be the stop token.

        # outputs.hidden_state is a tuple of len = n_generated_tokens.
        # The first hidden state is for the input tokens and is of shape
        #     (n_layers) x (batch_size, input_size, hidden_size).
        # (Note this includes the first generated token!)
        # The remaining hidden states are for the remaining generated tokens and is of shape
        #    (n_layers) x (batch_size, 1, hidden_size).

        # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size).
        # We do not get embeddings for input_data! We thus subtract the n_tokens_in_input from
        # token_stop_index to arrive at the right output.

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # If access idx is larger/equal.
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Get log_likelihoods.
        # outputs.scores are the logits for the generated token.
        # outputs.scores is a tuple of len = n_generated_tokens.
        # Each entry is shape (bs, vocabulary size).
        # outputs.sequences is the sequence of all tokens: input and generated.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        # Transition_scores[0] only contains the scores for the first generated tokens.

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']
        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()


# # Adding support for google/flan-t5 models

# from transformers import AutoModelForSeq2SeqLM # Import the correct class for T5 models

# class HuggingfaceModel(BaseModel):
#         """Hugging Face Model that can handle both Causal and Seq2Seq LM architectures."""

#         def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
#             if max_new_tokens is None:
#                 raise
#             self.max_new_tokens = max_new_tokens

#             if stop_sequences == 'default':
#                 stop_sequences = STOP_SEQUENCES

#             kwargs = {}
#             quantization_suffix = None
#             base_model_name = model_name

#             if base_model_name.endswith('-int8') or base_model_name.endswith('-8bit'):
#                 kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
#                 quantization_suffix = '-int8' if base_model_name.endswith('-int8') else '-8bit'
#                 logging.info(f"Preparing to load model in 8-bit mode.")
#             elif base_model_name.endswith('-4bit'):
#                 kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
#                 quantization_suffix = '-4bit'
#                 logging.info(f"Preparing to load model in 4-bit mode.")

#             if quantization_suffix:
#                 base_model_name = base_model_name[:-len(quantization_suffix)]

#             if 'llama' in base_model_name.lower():
#                 base = 'meta-llama' if ('Llama-2' in base_model_name or "Llama-3" in base_model_name) else 'huggyllama'
#                 if 'Llama-2' in base_model_name:
#                     base_model_name += '-hf'
#                 model_id = f"{base}/{base_model_name}"
#             elif 'mistral' in base_model_name.lower():
#                 model_id = f"mistralai/{base_model_name}"
#             elif 'falcon' in base_model_name.lower():
#                 model_id = f"tiiuae/{base_model_name}"
#             else:
#                 model_id = base_model_name

#             logging.info(f"Loading model from Hugging Face ID: {model_id}")
#             self.tokenizer = AutoTokenizer.from_pretrained(model_id)

#             if 't5' in model_id.lower():
#                 logging.info("Detected T5 architecture. Loading with AutoModelForSeq2SeqLM.")
#                 self.model = AutoModelForSeq2SeqLM.from_pretrained(
#                     model_id,
#                     device_map="auto",
#                     **kwargs
#                 )
#             else:
#                 logging.info("Detected Causal LM architecture. Loading with AutoModelForCausalLM.")
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_id,
#                     device_map="auto",
#                     trust_remote_code=True if 'falcon' in base_model_name.lower() else False,
#                     **kwargs
#                 )
            
#             self.model_name = model_name
#             self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
#             self.token_limit = 4096 if 'Llama-2' in model_name or "Llama-3" in model_name else 2048

#         def predict(self, input_data, temperature, return_full=False):
#             inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

#             if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
#                 if 'token_type_ids' in inputs:
#                     del inputs['token_type_ids']
#                 pad_token_id = self.tokenizer.eos_token_id
#             else:
#                 pad_token_id = None

#             if self.stop_sequences is not None:
#                 stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
#                     stops=self.stop_sequences,
#                     initial_length=len(inputs['input_ids'][0]),
#                     tokenizer=self.tokenizer)])
#             else:
#                 stopping_criteria = None

#             logging.debug('temperature: %f', temperature)
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=self.max_new_tokens,
#                     return_dict_in_generate=True,
#                     output_scores=True,
#                     output_hidden_states=True,
#                     temperature=temperature,
#                     do_sample=True,
#                     stopping_criteria=stopping_criteria,
#                     pad_token_id=pad_token_id,
#                 )

#             full_answer = self.tokenizer.decode(
#                 outputs.sequences[0], skip_special_tokens=True)

#             if return_full:
#                 return full_answer

#             if full_answer.startswith(input_data):
#                 input_data_offset = len(input_data)
#             else:
#                 input_data_offset = 0

#             answer = full_answer[input_data_offset:]
#             sliced_answer = answer.strip()
            
#             transition_scores = self.model.compute_transition_scores(
#                 outputs.sequences, outputs.scores, normalize_logits=True)
#             log_likelihoods = [score.item() for score in transition_scores[0]]
            
#             last_hidden_state = outputs.hidden_states[-1][-1] if outputs.hidden_states else None
#             last_token_embedding = last_hidden_state[:, -1, :].cpu() if last_hidden_state is not None else None
            
#             return sliced_answer, log_likelihoods, last_token_embedding

#         def get_p_true(self, input_data):
#             """Get the probability of the model anwering A (True) for the given input."""

#             input_data += ' A'
#             tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']
#             target_ids_true = tokenized_prompt_true.clone()
#             target_ids_true[0, :-1] = -100

#             with torch.no_grad():
#                 model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

#             loss_true = model_output_true.loss

#             return -loss_true.item()
    
# # End of T5 modifications


# Modification for better quantization support and model id handling

"""Implement HuggingfaceModel models with enhanced quantization support."""
import copy
import logging
from collections import Counter
import torch
import gc

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logging.warning("bitsandbytes not available. Quantization features will be limited.")

from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to(device) for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise ValueError("match_on must be 'text' or 'tokens'")
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""
    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class QuantizationConfig:
    """Helper class to manage quantization configurations."""
    
    @staticmethod
    def get_4bit_config(compute_dtype=torch.float16, quant_type="nf4", use_double_quant=True):
        """Get 4-bit quantization config with optimized settings."""
        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes is required for 4-bit quantization")
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,  # "nf4" or "fp4"
            bnb_4bit_use_double_quant=use_double_quant,  # Nested quantization for memory efficiency
        )
    
    @staticmethod
    def get_8bit_config():
        """Get 8-bit quantization config."""
        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes is required for 8-bit quantization")
        
        return BitsAndBytesConfig(load_in_8bit=True)
    
    @staticmethod 
    def parse_quantization_from_name(model_name):
        """Parse quantization settings from model name and return config."""
        quantization_config = None
        original_name = model_name
        
        # Check for different quantization suffixes
        if model_name.endswith('-4bit') or model_name.endswith('-4b'):
            quantization_config = QuantizationConfig.get_4bit_config()
            model_name = model_name.replace('-4bit', '').replace('-4b', '')
        elif model_name.endswith('-8bit') or model_name.endswith('-8b'):
            quantization_config = QuantizationConfig.get_8bit_config()  
            model_name = model_name.replace('-8bit', '').replace('-8b', '')
        elif model_name.endswith('-4bit-fp4'):
            quantization_config = QuantizationConfig.get_4bit_config(quant_type="fp4")
            model_name = model_name.replace('-4bit-fp4', '')
        elif model_name.endswith('-4bit-nf4'):
            quantization_config = QuantizationConfig.get_4bit_config(quant_type="nf4")
            model_name = model_name.replace('-4bit-nf4', '')
        elif model_name.endswith('-int4'):
            quantization_config = QuantizationConfig.get_4bit_config()
            model_name = model_name.replace('-int4', '')
        elif model_name.endswith('-int8'):
            quantization_config = QuantizationConfig.get_8bit_config()
            model_name = model_name.replace('-int8', '')
            
        return model_name, quantization_config


def get_available_memory():
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        available = total_memory - allocated
        return available, total_memory
    return 0, 0


def optimize_memory_settings(model_size_estimate_gb):
    """Get optimized memory settings based on model size and available memory."""
    available_mem, total_mem = get_available_memory()
    
    logging.info(f"Available GPU memory: {available_mem:.2f}GB / {total_mem:.2f}GB")
    logging.info(f"Estimated model size: {model_size_estimate_gb:.2f}GB")
    
    # Conservative memory allocation (leave 2GB buffer)
    max_memory_gb = max(available_mem - 2, total_mem * 0.8)
    
    memory_settings = {
        'max_memory': {0: f'{int(max_memory_gb)}GIB'} if torch.cuda.is_available() else None,
        'low_cpu_mem_usage': True,
        'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    
    return memory_settings


class HuggingfaceModel(BaseModel):
    """Enhanced Hugging Face Model with better quantization support."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None, 
                 quantization_config=None, memory_efficient=True, trust_remote_code=False):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be specified")
        
        self.max_new_tokens = max_new_tokens
        self.memory_efficient = memory_efficient
        
        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        # Clear GPU cache before loading
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Parse quantization from model name if not explicitly provided
        if quantization_config is None:
            model_name, quantization_config = QuantizationConfig.parse_quantization_from_name(model_name)
        
        self.original_model_name = model_name
        self.quantization_config = quantization_config
        
        # Log quantization status
        if quantization_config is not None:
            if hasattr(quantization_config, 'load_in_4bit') and quantization_config.load_in_4bit:
                logging.info("Using 4-bit quantization")
            elif hasattr(quantization_config, 'load_in_8bit') and quantization_config.load_in_8bit:
                logging.info("Using 8-bit quantization")
        else:
            logging.info("No quantization applied")

        try:
            if 'llama' in model_name.lower():
                self._load_llama_model(model_name, quantization_config, trust_remote_code)
            elif 'mistral' in model_name.lower():
                self._load_mistral_model(model_name, quantization_config, trust_remote_code)
            elif 'falcon' in model_name.lower():
                self._load_falcon_model(model_name, quantization_config, trust_remote_code)
            else:
                # Generic loading for other models
                self._load_generic_model(model_name, quantization_config, trust_remote_code)
                
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            # Try fallback with different quantization
            if quantization_config is not None:
                logging.info("Attempting fallback without quantization...")
                try:
                    if 'llama' in model_name.lower():
                        self._load_llama_model(model_name, None, trust_remote_code)
                    elif 'mistral' in model_name.lower():
                        self._load_mistral_model(model_name, None, trust_remote_code)
                    elif 'falcon' in model_name.lower():
                        self._load_falcon_model(model_name, None, trust_remote_code)
                    else:
                        self._load_generic_model(model_name, None, trust_remote_code)
                except Exception as fallback_e:
                    raise RuntimeError(f"Failed to load model even without quantization: {fallback_e}")
            else:
                raise

        self.model_name = model_name
        if self.tokenizer.eos_token is None:
            # Some models don't have eos_token, use pad_token or add one
            if self.tokenizer.pad_token is not None:
                self.tokenizer.eos_token = self.tokenizer.pad_token
            else:
                self.tokenizer.add_special_tokens({'eos_token': '</s>'})
                
        self.stop_sequences = (stop_sequences or []) + [self.tokenizer.eos_token]
        self.token_limit = 4096 if 'Llama-2' in model_name or "Llama-3" in model_name else 2048
        
        # Log final memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logging.info(f"Model loaded. GPU memory allocated: {allocated:.2f}GB")

    def _get_model_kwargs(self, quantization_config, trust_remote_code):
        """Get common model loading kwargs."""
        # Estimate model size (rough approximation)
        size_estimates = {
            '7b': 7, '8b': 8, '13b': 13, '30b': 30, '65b': 65, '70b': 70,
        }
        
        model_size_gb = 7  # default
        for size, gb in size_estimates.items():
            if size in self.original_model_name.lower():
                model_size_gb = gb
                break
        
        # Get optimized memory settings
        memory_settings = optimize_memory_settings(model_size_gb)
        
        kwargs = {
            'device_map': 'auto',
            'trust_remote_code': trust_remote_code,
            **memory_settings
        }
        
        if quantization_config is not None:
            kwargs['quantization_config'] = quantization_config
            
        return kwargs

    def _load_llama_model(self, model_name, quantization_config, trust_remote_code):
        """Load Llama models with enhanced quantization support."""
        if 'Llama-2' in model_name or "Llama-3" in model_name:
            base = 'meta-llama'
            if 'Llama-2' in model_name and not model_name.endswith('-hf'):
                model_name = model_name + '-hf'
        else:
            base = 'huggyllama'

        model_id = f"{base}/{model_name}"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token_type_ids=None,
            clean_up_tokenization_spaces=False
        )
        
        # Set pad_token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = self._get_model_kwargs(quantization_config, trust_remote_code)
        
        # Handle large models specially
        llama65b = '65b' in model_name and base == 'huggyllama'
        llama2_70b = '70b' in model_name and base == 'meta-llama'

        if llama2_70b or llama65b:
            if quantization_config is not None:
                logging.warning("Loading large model with quantization - this may take time")
            self._load_large_llama_model(model_id, kwargs)
        else:
            # Standard loading for smaller models
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def _load_large_llama_model(self, model_id, kwargs):
        """Load large Llama models with custom device mapping."""
        try:
            # Try direct loading first (works better with quantization)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except Exception as e:
            logging.info(f"Direct loading failed: {e}. Trying advanced loading...")
            
            # Fallback to the original complex loading method
            path = snapshot_download(
                repo_id=model_id,
                allow_patterns=['*.json', '*.model', '*.safetensors'],
                ignore_patterns=['pytorch_model.bin.index.json']
            )

            config = AutoConfig.from_pretrained(model_id)
            with accelerate.init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(config)
            self.model.tie_weights()
            
            available_mem, _ = get_available_memory()
            max_mem = int(available_mem * 0.4 * 1024**3)  # 40% of available memory per device
            
            device_map = accelerate.infer_auto_device_map(
                self.model.model,
                max_memory={0: max_mem, 1: max_mem} if torch.cuda.device_count() > 1 else {0: max_mem * 2},
                dtype='float16'
            )
            device_map = remove_split_layer(device_map)
            full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
            full_model_device_map["lm_head"] = 0

            self.model = accelerate.load_checkpoint_and_dispatch(
                self.model, path, device_map=full_model_device_map,
                dtype='float16', skip_keys='past_key_values'
            )

    def _load_mistral_model(self, model_name, quantization_config, trust_remote_code):
        """Load Mistral models with enhanced quantization support."""
        model_id = f'mistralai/{model_name}'
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token_type_ids=None,
            clean_up_tokenization_spaces=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = self._get_model_kwargs(quantization_config, trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def _load_falcon_model(self, model_name, quantization_config, trust_remote_code):
        """Load Falcon models with enhanced quantization support."""
        model_id = f'tiiuae/{model_name}'
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token_type_ids=None,
            clean_up_tokenization_spaces=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = self._get_model_kwargs(quantization_config, trust_remote_code)
        # Falcon requires trust_remote_code
        kwargs['trust_remote_code'] = True
        
        # If no quantization specified, use 8-bit for Falcon by default (they're large)
        if quantization_config is None:
            kwargs['quantization_config'] = QuantizationConfig.get_8bit_config()
            
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def _load_generic_model(self, model_name, quantization_config, trust_remote_code):
        """Generic model loading for other architectures."""
        # Try to infer the full model path
        if '/' not in model_name:
            # Try common patterns
            possible_prefixes = [
                'microsoft/', 'google/', 'facebook/', 'huggingface/', 
                'EleutherAI/', 'bigscience/', 'stabilityai/'
            ]
            
            model_id = model_name
            for prefix in possible_prefixes:
                try:
                    test_id = f'{prefix}{model_name}'
                    AutoTokenizer.from_pretrained(test_id)
                    model_id = test_id
                    break
                except:
                    continue
        else:
            model_id = model_name
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token_type_ids=None,
            clean_up_tokenization_spaces=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = self._get_model_kwargs(quantization_config, trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def predict(self, input_data, temperature, return_full=False):
        """Enhanced prediction with better error handling."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(input_data, return_tensors="pt").to(device)

        # Clean up token_type_ids if present (not needed for most modern models)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
            
        pad_token_id = self.tokenizer.eos_token_id

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        
        generation_kwargs = {
            **inputs,
            'max_new_tokens': self.max_new_tokens,
            'return_dict_in_generate': True,
            'output_scores': True,
            'output_hidden_states': True,
            'temperature': temperature,
            'do_sample': True,
            'pad_token_id': pad_token_id,
        }
        
        if stopping_criteria is not None:
            generation_kwargs['stopping_criteria'] = stopping_criteria
            
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        if len(outputs.sequences[0]) > self.token_limit:
            logging.warning(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            logging.warning(f"Full answer doesn't start with input_data. Setting offset to 0")
            logging.debug(f"Full answer: {full_answer[:100]}...")
            logging.debug(f"Input data: {input_data}")
            input_data_offset = 0

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            
            # Check if stop words are properly removed
            for stop in self.stop_sequences:
                if stop in sliced_answer:
                    logging.warning(f'Stop word "{stop}" still present in answer')

        # Remove whitespaces from answer
        sliced_answer = sliced_answer.strip()

        # Get token counts and embeddings (rest of the original logic)
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. Taking next token for embeddings.')
            n_generated = 1

        # Get hidden states
        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning('Taking first and only generation for hidden states')
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            logging.warning('Taking last state because n_generated is too large')
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Get last layer and token embedding
        last_layer = last_input[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Get log likelihoods
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) > n_generated:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.info('Generation stopped due to max_token limit')

        if len(log_likelihoods) == 0:
            raise ValueError("No log likelihoods computed")

        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model answering A (True) for the given input."""
        device = next(self.model.parameters()).device
        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to(device)['input_ids']

        target_ids_true = tokenized_prompt_true.clone()
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss
        return -loss_true.item()

    def cleanup(self):
        """Clean up model from memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.info("Model cleaned up from memory")


# Utility functions for easy model loading in Kaggle
def get_kaggle_optimized_model(model_choice="mistral-7b"):
    """
    Get a model optimized for Kaggle environment constraints.
    
    Args:
        model_choice: One of "mistral-7b", "llama2-7b", "llama3-8b"
    """
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Model configurations optimized for Kaggle
    configs = {
        "mistral-7b": {
            "model_name": "Mistral-7B-v0.1-4bit",
            "max_new_tokens": 256,
            "description": "Most memory efficient, good performance"
        },
        "llama2-7b": {
            "model_name": "Llama-2-7b-chat-hf-8bit", 
            "max_new_tokens": 256,
            "description": "Well-tested, reliable"
        },
        "llama3-8b": {
            "model_name": "Llama-3-8B-8bit",
            "max_new_tokens": 200,
            "description": "Latest, but uses more memory"
        }
    }
    
    if model_choice not in configs:
        raise ValueError(f"Choose from: {list(configs.keys())}")
    
    config = configs[model_choice]
    print(f"Loading {model_choice}: {config['description']}")
    
    try:
        model = HuggingfaceModel(
            model_name=config["model_name"],
            max_new_tokens=config["max_new_tokens"],
            stop_sequences='default'
        )
        print(f"✅ Successfully loaded {model_choice}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_choice}: {e}")
        return None


def check_gpu_memory():
    """Check current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print("CUDA not available")