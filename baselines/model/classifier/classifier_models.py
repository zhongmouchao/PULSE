import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, device: torch.device | None = None):
        """
        One hidden layer, with no dropout and no batch norm.

        Args:
            input_dim (int): Input dimension (number of features)
            hidden_dim (int): Hidden layer dimension (default: 16)
            device (torch.device): Device to place the model on
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.hidden_layer(x))
        return self.sigmoid(self.output_layer(hidden)) 


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, device: torch.device):
        """
        Simple logistic regression classifier (no hidden layer)
        
        Args:
            input_dim (int): Input dimension (number of features)
            device (torch.device): Device to place the model on
        """
        super().__init__()
        self.input_dim = input_dim
        
        # Single linear layer for binary classification
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Move model to device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Binary prediction probabilities of shape (batch_size, 1)
        """
        output = self.linear(x)
        output = self.sigmoid(output)
        
        return output

class WideDeepNetwork(nn.Module):
    def __init__(self, input_dim, wide_dim= 64, deep_dims=[32, 16], output_dim=1, dropout_prob=0.3, device: torch.device | None = None):
        super(WideDeepNetwork, self).__init__()
        if wide_dim is None:
            wide_dim = input_dim
            
        # Wide part (linear transformation)
        self.wide_layer = nn.Linear(input_dim, wide_dim)
        
        # Deep part
        self.deep_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        dims = [input_dim] + deep_dims
        for i in range(len(dims) - 1):
            self.deep_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))
            self.dropouts.append(nn.Dropout(dropout_prob))
        
        # Combine wide and deep
        combined_dim = wide_dim + deep_dims[-1]
        self.output_layer = nn.Linear(combined_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        # Wide part
        wide_out = self.wide_layer(x)
        
        # Deep part
        deep_out = x
        for i, layer in enumerate(self.deep_layers):
            deep_out = layer(deep_out)
            deep_out = self.batch_norms[i](deep_out)
            deep_out = F.relu(deep_out)
            deep_out = self.dropouts[i](deep_out)
        
        # Combine and output
        combined = torch.cat([wide_out, deep_out], dim=-1)
        output = torch.sigmoid(self.output_layer(combined))
        return output
    


class WideDeepTransformer(nn.Module):
    def __init__(self, input_dim, device: torch.device, wide_dim=16, transformer_dim=16, n_heads=4, n_layers=2, output_dim=1, dropout_prob=0.3, attn_dropout_prob=0.1 ):
        super(WideDeepTransformer, self).__init__()
        self.to(device)
        if wide_dim is None:
            wide_dim = input_dim
        
        # Wide part (linear transformation)
        self.wide_layer = nn.Linear(input_dim, wide_dim)
        
        # --- NEW: feature ID embeddings for numerical features ---
        self.feature_embeddings = nn.Embedding(input_dim, transformer_dim)

        # --- NEW: value projection (turns scalar value into vector) ---
        self.value_proj = nn.Linear(1, transformer_dim)

        # Deep part replaced with Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=transformer_dim * 2,
            dropout=dropout_prob,
            activation='relu',
            batch_first=True
        )
        # Attention dropout
        if hasattr(encoder_layer.self_attn, "dropout"):
            encoder_layer.self_attn.dropout = attn_dropout_prob
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(transformer_dim)

        # Combine wide and deep
        combined_dim = wide_dim + transformer_dim
        self.output_layer = nn.Linear(combined_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        # Wide part
        wide_out = self.wide_layer(x)
        
        # Deep part (FT-Transformer style)
        batch_size, num_features = x.shape

        # Expand numerical features: (batch, num_features, 1)
        x_expanded = x.unsqueeze(-1)

        # Project values into embedding space
        value_emb = self.value_proj(x_expanded)    # (batch, num_features, transformer_dim)

        # Add feature ID embeddings
        feature_ids = torch.arange(num_features, device=x.device)
        feature_emb = self.feature_embeddings(feature_ids)   # (num_features, transformer_dim)
        feature_emb = feature_emb.unsqueeze(0).expand(batch_size, -1, -1)  # match batch

        # Final tokens = value embedding + feature embedding
        tokens = value_emb + feature_emb

        # Transformer
        trans_out = self.transformer(tokens)       # (batch, num_features, dim)
        deep_out = trans_out.mean(dim=1)           # aggregate features
        deep_out = self.layer_norm(deep_out)
        deep_out = self.dropout(deep_out)
        
        # Combine and output
        combined = torch.cat([wide_out, deep_out], dim=-1)
        output = torch.sigmoid(self.output_layer(combined))
        return output

class TransformerClassifier(nn.Module):
    """
    Lightweight transformer classifier for PU likelihood setting.

    Changes:
        1. Smaller transformer capacity.
        2. CLS token pooling.
        3. Linear residual path.
    """

    def __init__(
        self,
        input_dim,
        device: torch.device,
        d_model=8,
        n_heads=2,
        n_transformer_layers=1,
        ff_dim=16,
        mlp_hidden_dim=8,
        dropout_prob=0.4,
        output_dim=1,
    ):
        super(TransformerClassifier, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.d_model = d_model

        self.feature_embedding = nn.Linear(1, d_model)

        self.feature_tokens = nn.Parameter(
            torch.randn(1, input_dim, d_model)
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=n_transformer_layers,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_dim, output_dim),
        )

        # Logistic-regression-like residual path
        # self.linear_head = nn.Linear(input_dim, output_dim)

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.to(self.device)

        x_original = x
        batch_size = x.size(0)

        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)

        x = x + self.feature_tokens

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        h = self.transformer_encoder(x)

        h_cls = h[:, 0, :]

        logits_transformer = self.mlp_head(h_cls)
        # logits_linear = self.linear_head(x_original)
        logits = logits_transformer
        # logits = logits_transformer + logits_linear

        probs = torch.sigmoid(logits)

        return probs

# class TransformerClassifier(nn.Module):
#     """
#     Transformer-based neural network for binary classification.

#     This model:
#         1. Takes a vector input x ∈ R^d
#         2. Treats each feature as a token
#         3. Uses Transformer encoder layers to learn feature interactions
#         4. Outputs probability p(y=1|x)

#     IMPORTANT:
#     ----------
#     This version RETURNS probabilities directly
#     using sigmoid at the end.
#     """


#     def __init__(
#         self,
#         input_dim,
#         device: torch.device,
#         d_model=32,
#         n_heads=2,
#         n_transformer_layers=1,
#         ff_dim=128,
#         mlp_hidden_dim=32,
#         dropout_prob=0.1,
#         output_dim=1,
#     ):
#         """
#         Parameters
#         ----------
#         input_dim : int
#             Number of input features.

#         d_model : int
#             Embedding dimension used inside transformer.

#         n_heads : int
#             Number of self-attention heads.

#         n_transformer_layers : int
#             Number of transformer encoder blocks.

#         ff_dim : int
#             Hidden dimension of feedforward network
#             inside transformer blocks.

#         mlp_hidden_dim : int
#             Hidden dimension of final MLP head.

#         dropout_prob : float
#             Dropout probability.

#         output_dim : int
#             Output dimension.
#             Usually 1 for binary classification.
#         """

#         super(TransformerClassifier, self).__init__()
#         self.device = device
#         self.input_dim = input_dim
#         self.d_model = d_model

#         self.feature_embedding = nn.Linear(1, d_model)

#         self.feature_tokens = nn.Parameter(
#             torch.randn(1, input_dim, d_model)
#         )

#         transformer_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=ff_dim,
#             dropout=dropout_prob,
#             activation="gelu",
#             batch_first=True,
#         )

#         # Stack multiple transformer layers
#         self.transformer_encoder = nn.TransformerEncoder(
#             transformer_layer,
#             num_layers=n_transformer_layers,
#         )
#         self.mlp_head = nn.Sequential(

#             nn.LayerNorm(d_model),

#             nn.Linear(d_model, mlp_hidden_dim),

#             nn.GELU(),

#             nn.Dropout(dropout_prob),

#             nn.Linear(mlp_hidden_dim, output_dim),
#         )

#         # Initialize weights
#         self._initialize_weights()
#         self.to(self.device)


#     def _initialize_weights(self):
#         """
#         Initialize weights using Xavier initialization.
#         """

#         for m in self.modules():

#             if isinstance(m, nn.Linear):

#                 nn.init.xavier_uniform_(m.weight)

#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)


#     def forward(self, x):
#         """
#         Forward pass.

#         Parameters
#         ----------
#         x : torch.Tensor

#             Shape:
#                 (batch_size, input_dim)

#         Returns
#         -------
#         probs : torch.Tensor

#             Shape:
#                 (batch_size, output_dim)

#             Predicted probabilities in [0, 1].
#         """
#         x = x.to(self.device)
#         x = x.unsqueeze(-1)
#         x = self.feature_embedding(x)
#         x = x + self.feature_tokens
#         h = self.transformer_encoder(x)
#         h_pool = h.mean(dim=1)
#         logits = self.mlp_head(h_pool)
#         probs = torch.sigmoid(logits)

#         return probs


class NAM(nn.Module):
    """Neural additive model. Each position has its own network on a chemical triplet.

    The peptide probability is a sigmoid of the scaled sum of per-site scores.
    input_dim must be divisible by 3, which is the CHEM feature layout.
    """

    def __init__(self, input_dim, hidden_dim=4, device=None, **kw):
        super().__init__()
        if input_dim % 3 != 0:
            raise ValueError(
                f'NAM expects chemical triplet features, got input_dim={input_dim}'
            )
        self.input_dim = input_dim
        self.n_sites = input_dim // 3
        self.hidden_dim = hidden_dim
        self.W1 = nn.Parameter(torch.empty(self.n_sites, 3, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.n_sites, hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(self.n_sites, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.n_sites))
        nn.init.normal_(self.W1, std=math.sqrt(2.0 / (3 + hidden_dim)))
        self.scale = 1.0 / math.sqrt(self.n_sites)
        if device is not None:
            self.to(device)

    def forward(self, x):
        sites = x.view(-1, self.n_sites, 3)
        hidden = torch.relu(torch.einsum('bsi,sih->bsh', sites, self.W1) + self.b1)
        contrib = torch.einsum('bsh,sh->bs', hidden, self.W2) + self.b2
        return torch.sigmoid(contrib.sum(dim=1, keepdim=True) * self.scale)