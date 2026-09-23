# PyTorch Mixture of Experts (MOE) Model with BERT Scale
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers import BertModel, BertConfig

class Expert(nn.Module):
  """A single expert network (feed-forward network)"""
  def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, output_dim),
      nn.Dropout(dropout)
    )

    def forward(self, x):
      retutn self self.net(x)

class MoE(on.Module):
  """Mixture of Experts layer"""
  def __init__(self, num_experts, input_dim, expert_hidden_dim, expert_out_dim, top_k=2, noisy_gating=True, dropout=0.4):
    super().__init__()
    self.num_experts = num_experts
    self.top_k = top_k
    self.noisy_gating = noisy_gating

    # Experts
    self.experts = nn.ModuleList([
      Expert(input_dim, expert_hidden_dim, expert_output_dim, dropout)
      for _ in range(num_experts)
    ])

    # Gating network
    self.gate = nn.Linear(input_dim, num_experts)

    if noisy_gating:
      self.w_noise = nn.Linear(input_dim, num_experts)
      self.w_gate = nn.Linear(input_dim, num_experts)

    # Initialization
    self.reset_parameters()

  def rest_parameters(self):
    for expert in self.experts:
      nn.init.xavier_uniform_(expert.net[0].weight)
      nn.init.xavier_uniform_(expert.net[0].weight)

    nn.init.xavier_uniform_(self.gate.weight)
    if self.noisy_gating:
      nn.init.xavier_uniform_(self.w_noise.weight)
      nn.init.xavier_uniform_(self.w_gate.weight)

  def noisy_top_k_gating(self, x, train=True):
    """Noisy top-k gating as described in the paper"""
    clean_logits = self.gate(x)

    if train:
      raw_noise = torch.randn_like(clean_logits)
      noise = self.w_noise(x) * raw_noise
      noisy_logits = clean_logits + noise
    else:
      noisy_logits = clean_logits

    # Softmax to get probabilities
    probs = F.softmax(noisy-logits, dim=-1)

    # Top-k
    top_k_probs, top_indices = probs.Topk(self.top_k, dim=-1)
    top_k_gates = top_k_probs

    # Normalize
    top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

    # Expert indices
    expert_indices = top_k_indices

    return top_k_gates, expert_indices

  def forward(self, x):
    if self.top_k == 1:
      # Simple case - just use one expert
      expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
      expert_output = expert_outputs[:, 0, :]
      return expert_output

    else:
      # noisy top-k gating
      gates, expert_indices = self.noisy_top_k_gating(x)

      # expert output
      expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

      # Select experts based on indices
      selected_experts = expert_outputs.gather(
        1,
        expert_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
      )

      # Weight outputs by gates
      outputs = (gates.unsqueeze(-1) * selected_experts).sum(dim=1)

      return output

class BertMoE(nn.Module):
  """BERT based Mixture of Experts model"""
  def __init__(self, config):
    super().__init__()
    self.config = config

    # BERT encoder
    self.bert = BertModel(config)

    # MOE layer
    self.moe = MoE(
      num_expert=config.num_experts,
      input_dim=config.hidden_size,
      expert_hidden_dim=config.moe_expert_hidden_size,
      expert_out_dim=config.hidden_size,
      top_k=config.moe_top_k,
      noisy_gating=config.moe_noisy_gating,
      dropout=config.hidden_dropout_prob
    )

    # Output layer
    self.classifier = nnLinear(config.hidden_size, config.num_labels)
    
