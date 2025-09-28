import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

def stable_mvm(log_W, log_x):
    log_x_max = torch.amax(log_x)
    log_x_norm = log_x - log_x_max
    x_norm = torch.exp(log_x_norm)

    log_W_max = torch.amax(log_W, dim=1, keepdim=True)
    log_W_norm = log_W - log_W_max
    W_norm = torch.exp(log_W_norm)

    z_norm = W_norm @ x_norm

    return torch.log(z_norm) + log_x_max + torch.squeeze(log_W_max, dim=-1)

class HMM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_size, vocab_size, eos_token_id, sep_token_id=None):
        super().__init__()

        alpha_exp = torch.softmax(torch.randn(hidden_size, hidden_size), dim=-1)
        beta = torch.log_softmax(torch.randn(hidden_size, vocab_size), dim=-1)
        gamma_exp = torch.softmax(torch.randn(1, hidden_size), dim=-1)

        if sep_token_id is not None:
            ################# SEP TOKEN INITIALIZATION #################
            beta[-1, sep_token_id] = 1e10
            beta[:-1, sep_token_id] = -1e10
            beta = torch.log_softmax(beta, dim=-1)
            alpha_exp[-1, -1] = 1e-10
            ################# SEP TOKEN INITIALIZATION #################
        else:
            ################# EOS TOKEN INITIALIZATION #################
            beta[-1, eos_token_id] = 1e10
            beta[:-1, eos_token_id] = -1e10
            beta = torch.log_softmax(beta, dim=-1)
            alpha_exp[-1, :] = 1e-10
            alpha_exp[-1, -1] = 1.0
            ################# EOS TOKEN INITIALIZATION #################

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=True)
        self.beta = nn.Parameter(beta, requires_grad=True)
        self.gamma_exp = nn.Parameter(gamma_exp, requires_grad=True)

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id


        # ------------------- Set #1 of weights -------------------------
        self.register_buffer('weights_tensor', torch.zeros(vocab_size, dtype=torch.float32))  
        self.register_buffer('exp_weights', torch.ones(vocab_size, dtype=torch.float32))      
        self.register_buffer('weighted_beta', torch.zeros(hidden_size, vocab_size, dtype=torch.float32))

        # Initialize set #1 to something by default (optional)
        self.set_weights(weights_tensor=torch.zeros(vocab_size))

    @property
    def pi(self):
        """Initial state distribution (gamma_exp)"""
        return self.gamma_exp.squeeze(0)  # Convert from (1, H) to (H)
    
    @property  
    def log_B(self):
        """Log emission probabilities (beta)"""
        return self.beta  # Already in log space
    
    @property
    def w(self):
        """Exponentiated weights for toxicity"""
        return self.exp_weights

    def set_weights(self, weights_tensor: torch.Tensor):
        """
        Set the weights and update weighted_beta accordingly.
        """
        if weights_tensor.shape != (self.vocab_size,):
            raise ValueError(
                f"weights_tensor must have shape ({self.vocab_size},), but got {weights_tensor.shape}"
            )

        # Update weights_tensor and exp_weights
        self.weights_tensor.copy_(weights_tensor)  
        self.exp_weights.copy_(torch.exp(weights_tensor))  

        # Update weighted_beta: P(x|s) * exp{w(x)}
        P_x_given_s = torch.exp(self.beta)  # (H, V)
        weighted_beta = P_x_given_s * self.exp_weights.unsqueeze(0)  # (H, V)
        self.weighted_beta.copy_(weighted_beta)  # (H, V)

    def forward_step(self, current_z: torch.Tensor, next_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute one forward step of the HMM given current hidden state distribution and next tokens.
        
        Args:
            current_z: Current hidden state distribution, shape (B, H)
            next_tokens: Next observed tokens, shape (B,)
        
        Returns:
            Updated hidden state distribution, shape (B, H)
        """
        # Transition: P(z_t+1 | z_t) using alpha_exp matrix
        # current_z: (B, H), alpha_exp: (H, H) -> (B, H, H)
        transition_probs = current_z.unsqueeze(2) * self.alpha_exp.unsqueeze(0)  # (B, H, H)
        
        # Sum over previous states to get marginal: P(z_t+1)
        next_z_marginal = transition_probs.sum(dim=1)  # (B, H)
        
        # Emission: P(x_t+1 | z_t+1) using beta matrix  
        # beta: (H, V), next_tokens: (B,) -> (B, H)
        emission_logprobs = self.beta[:, next_tokens].T  # (B, H)
        emission_probs = torch.exp(emission_logprobs)    # (B, H)
        
        # Apply emission probabilities
        updated_z = next_z_marginal * emission_probs  # (B, H)
        
        # Normalize to ensure it's a valid probability distribution
        updated_z = updated_z / (updated_z.sum(dim=1, keepdim=True) + 1e-12)
        
        return updated_z

    def compute_forward_probability(self, input_ids):
        batch_size, m = input_ids.size()
        x = input_ids[:, 0]
        emission = self.beta[:, x].transpose(0, 1) 
        gamma = torch.log(self.gamma_exp)
        alpha_prev = gamma + emission

        for t in range(1, m):
            alpha_prev = torch.vmap(stable_mvm, in_dims=(None, 0))(torch.log(self.alpha_exp + 1e-12).T, alpha_prev)
            x_t = input_ids[:, t]
            emission = self.beta[:, x_t].transpose(0, 1)
            alpha_prev = alpha_prev + emission

        return alpha_prev

    def compute_backward_expectation_for_weights(self, T: int, weighted_beta: torch.Tensor) -> torch.Tensor:
        """
        Reusable routine to compute B(t, s) for any given weighted_beta.
        B[t, s] = E[exp{ sum_{i=t+1}^T w(x_i) }] for z_t = s.
        """
        device = self.alpha_exp.device
        hidden_states = self.alpha_exp.shape[0]

        B = torch.ones((T, hidden_states), dtype=torch.float32, device=device)

        # Precompute sum_x P(x|s') * exp{w(x)} for each s'
        weighted_emission_sum = torch.sum(weighted_beta, dim=1)  # (H,)

        # Backward recursion
        for t in reversed(range(T - 1)):
            temp = weighted_emission_sum * B[t + 1, :]  # (H,)
            # B(t, s) = Σ_{s'} [ α_exp[s, s'] * temp[s'] ]
            B[t, :] = torch.matmul(self.alpha_exp, temp)

        return B

    def compute_backward_expectation(self, T: int) -> torch.Tensor:
        """
        Compute backward expectation using the *first* set of weights (weighted_beta).
        """
        return self.compute_backward_expectation_for_weights(T, self.weighted_beta)

