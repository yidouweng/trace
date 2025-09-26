import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Tuple, Any

from src.hmm import stable_mvm

@torch.compile
def logit_adjustment(
    log_alpha_prev: torch.Tensor,          # (B, H) - current HMM state distribution  
    log_A: torch.Tensor,                   # (1, H, H) - HMM transition matrix
    log_B: torch.Tensor,                   # (H, V) - HMM emission matrix
    expectation_zm: torch.Tensor,          # (H,) - backward expectation cache for current step
    product_generated_toxicity: torch.Tensor,  # (B,) - cumulative toxicity product 
    exp_weights: torch.Tensor,             # (V,) - token weights exp(w(x))
    scores: torch.Tensor,                  # (B, V) - original language model logits
    a: float,                              # guidance strength
    epsilon: float = 1e-12                 # numerical stability
) -> torch.Tensor:
    """
    Exact implementation of CTRL-G adjustment logic with torch.compile optimization.
    
    This function implements exactly the same computation as in ctrlg.py lines 137-153:
    1. Compute forward transition probabilities
    2. Compute token-specific hidden state posteriors  
    3. Compute expected future toxicity for each token
    4. Apply current toxicity product and token weights
    5. Apply sigmoid-logit scaling 
    6. Re-weight language model probabilities and normalize
    """
    # Step 1: Compute forward transition probabilities
    log_p_zm_x_less_m = torch.vmap(stable_mvm, in_dims=(None, 0))(log_A.squeeze(0).T, log_alpha_prev)  # (B, H)

    # Step 2: Compute expected future toxicity for each token
    log_p_x = torch.vmap(stable_mvm, in_dims=(None, 0))(log_B.T, log_p_zm_x_less_m)  # (B, V)
    log_expectation_zm_x_less_m = log_p_zm_x_less_m + torch.log(expectation_zm + epsilon)  # (B, H)
    log_expectation_xm = torch.vmap(stable_mvm, in_dims=(None, 0))(log_B.T, log_expectation_zm_x_less_m)  - log_p_x # (B, V)
    # Step 3: Apply current toxicity product and token weights
    log_expectation_xm += torch.log(product_generated_toxicity.unsqueeze(1) + epsilon) + torch.log(exp_weights + epsilon)
    
    # Step 4: Apply sigmoid-logit scaling
    expectation_xm = torch.exp(log_expectation_xm)
    expectation_xm = torch.clamp(expectation_xm, epsilon, 1 - epsilon)
    logit_expectation_xm = torch.log(expectation_xm / (1 - expectation_xm + epsilon) + epsilon)
    logit_adjusted = a * logit_expectation_xm
    expectation_xm_adjusted = torch.sigmoid(logit_adjusted)
    
    # Step 5: Re-weight language model probabilities and normalize
    p_lm = torch.softmax(scores, dim=-1)
    # print(f"p_lm: {p_lm}")
    p_adjusted = p_lm * expectation_xm_adjusted  
    p_adjusted = p_adjusted / (p_adjusted.sum(dim=-1, keepdim=True) + epsilon)
    adjusted_logits = torch.log(p_adjusted + epsilon)
    
    return adjusted_logits

class HmmGuidedLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 hmm_model: Any,
                 expectation_cache: List[torch.Tensor],
                 a: float = 1.0,
                 tokenizer: PreTrainedTokenizer | None = None,
                 epsilon: float = 1e-12):
        # Check for required HMM model attributes (support both exp_weights and exp_weights_1)
        required_attrs = ['alpha_exp', 'beta', 'compute_forward_probability']
        for attr in required_attrs:
            if not hasattr(hmm_model, attr):
                raise AttributeError(f"HMM model missing required attribute: {attr}")
        
        # Check which exp_weights attribute exists (support both naming conventions)
        if hasattr(hmm_model, 'exp_weights_1'):
            self.exp_weights_attr = 'exp_weights_1'
        elif hasattr(hmm_model, 'exp_weights'):
            self.exp_weights_attr = 'exp_weights'
        else:
            raise AttributeError("HMM model must have either 'exp_weights' or 'exp_weights_1' attribute")

        self.hmm_model = hmm_model
        self._model_device = hmm_model.alpha_exp.device
        
        self.expectation_cache = [ec.to(self._model_device) for ec in expectation_cache]
        self.a = a
        self.epsilon = epsilon
        self.tokenizer = tokenizer

        # Pre-compute commonly used tensors (exactly like CTRL-G)
        self.log_A = torch.log(self.hmm_model.alpha_exp.to(self._model_device) + self.epsilon).unsqueeze(0)
        self.log_B = self.hmm_model.beta.to(self._model_device)
        
        # Get exp_weights using the correct attribute name
        exp_weights_raw = getattr(self.hmm_model, self.exp_weights_attr)
        self.exp_weights = exp_weights_raw.squeeze(-1).to(self._model_device)

        # State tracking variables - these will be set during configure_for_prompts
        self.log_alpha_prev = None
        self.product_generated_toxicity = None
        self.prompt_len = -1
        self.is_configured_for_prompt = False
        self.generation_step = 0  # Track which generation step we're on

    def configure_for_prompts(self, prompt_batch: torch.Tensor):
        """Configure the processor for a new set of prompts."""
        if not self.hmm_model or not self.expectation_cache:
            raise RuntimeError("HMM model and expectation cache must be set before configuration.")
        
        self.prompt_len = prompt_batch.shape[1]
        self.is_configured_for_prompt = True
        self.generation_step = 0
        
        # Initialize HMM forward probabilities for the prompts (exactly like CTRL-G)
        self.log_alpha_prev = self.hmm_model.compute_forward_probability(prompt_batch)
        
        # Initialize product of generated toxicity (exactly like CTRL-G)
        if hasattr(self.hmm_model, 'intercept_1'):
            initial_product = torch.exp(self.hmm_model.intercept_1)
        else:
            initial_product = 1.0
        
        batch_size = prompt_batch.shape[0]
        self.product_generated_toxicity = torch.full((batch_size,), initial_product, 
                                                   device=self._model_device, dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Process logits to apply HMM-based toxicity guidance using TRACE rule.
        This implementation exactly matches the CTRL-G logic flow using torch.compile.
        """
        with torch.no_grad():
            # Check if processor has been configured
            if not self.is_configured_for_prompt:
                raise RuntimeError("HmmGuidedLogitsProcessor.configure_for_prompts() must be called before generation.")
            
            # ====================================================================
            # UPDATE STATE FIRST (if new tokens have been generated)
            # ====================================================================
            
            # Check if we have new tokens in input_ids and update state accordingly
            current_seq_len = input_ids.shape[1]
            expected_seq_len = self.prompt_len + self.generation_step
            
            if current_seq_len > expected_seq_len:
                # We have new tokens! Update state for the new tokens (exactly like CTRL-G)
                new_token_pos = expected_seq_len  # Position of the new token
                new_tokens = input_ids[:, new_token_pos]  # (batch_size,)
                
                # First, compute the forward transition for state update
                temp = self.log_alpha_prev.unsqueeze(2) + self.log_A  # (batch_size, H, H)
                log_p_zm_x_less_m = torch.logsumexp(temp, dim=1)  # (batch_size, H)
                
                # Update product of generated toxicity weights (exactly like CTRL-G)
                self.product_generated_toxicity *= self.exp_weights[new_tokens]
                
                # Update HMM forward probabilities (exactly like CTRL-G)
                emission = self.hmm_model.beta[:, new_tokens].transpose(0, 1)  # (batch_size, H)
                self.log_alpha_prev = log_p_zm_x_less_m + emission  # (batch_size, H)
                
                # Increment generation step
                self.generation_step += 1
            
            # Check bounds for expectation cache access
            if self.generation_step >= len(self.expectation_cache):
                # Beyond cache length, no guidance available
                return scores
            
            # ====================================================================
            # APPLY CTRL-G GUIDANCE USING COMPILED FUNCTION
            # ====================================================================
            
            # Get expectation cache for current step
            expectation_zm = self.expectation_cache[self.generation_step].to(self._model_device)  # (H,)
            
            # Apply the compiled CTRL-G adjustment logic 
            adjusted_logits = logit_adjustment(
                log_alpha_prev=self.log_alpha_prev,
                log_A=self.log_A,
                log_B=self.log_B,
                expectation_zm=expectation_zm,
                product_generated_toxicity=self.product_generated_toxicity,
                exp_weights=self.exp_weights,
                scores=scores,
                a=self.a,
                epsilon=self.epsilon
            )
            
            return adjusted_logits