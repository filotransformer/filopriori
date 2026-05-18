"""
Graph Attention Network v2 (GATv2) Layer

Implementação da GATv2 conforme proposto em:
"How Attentive are Graph Attention Networks?" (Brody et al., 2022)

Diferença chave de GAT: LeakyReLU aplicado APÓS projeção linear,
permitindo atenção dinâmica baseada em features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add


class GATv2Conv(nn.Module):
    """
    Graph Attention Network v2 convolution layer.

    Args:
        in_channels: Dimensão de entrada
        out_channels: Dimensão de saída por head
        num_heads: Número de attention heads
        concat: Se True, concatena heads; se False, faz média
        dropout: Taxa de dropout para attention weights
        add_self_loops: Se True, adiciona self-loops ao grafo
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        add_self_loops: bool = True,
        bias: bool = True,
        negative_slope: float = 0.2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope

        # Projeção linear compartilhada entre heads
        self.lin = nn.Linear(in_channels, num_heads * out_channels, bias=False)

        # Vetor de atenção (um por head)
        self.att = nn.Parameter(torch.Tensor(1, num_heads, out_channels))

        # Bias opcional
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]

        Returns:
            out: [num_nodes, num_heads * out_channels] se concat=True
                 [num_nodes, out_channels] se concat=False
        """
        N, H, C = x.size(0), self.num_heads, self.out_channels

        # Transformação linear
        x = self.lin(x).view(N, H, C)  # [N, H, C]

        # Adicionar self-loops
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        # Obter features de source e target nodes
        row, col = edge_index
        x_i = x[row]  # [num_edges, H, C]
        x_j = x[col]  # [num_edges, H, C]

        # GATv2: LeakyReLU ANTES da aplicação do vetor de atenção
        # Isso é a mudança chave de GAT → GATv2
        alpha = F.leaky_relu(x_i + x_j, self.negative_slope)  # [num_edges, H, C]
        alpha = (alpha * self.att).sum(dim=-1)  # [num_edges, H]

        # Softmax sobre vizinhos de cada node
        alpha = softmax(alpha, row, num_nodes=N)  # [num_edges, H]

        # Aplicar dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Agregação ponderada
        out = x_j * alpha.unsqueeze(-1)  # [num_edges, H, C]
        out = scatter_add(out, row, dim=0, dim_size=N)  # [N, H, C]

        # Concatenar ou fazer média dos heads
        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)

        # Adicionar bias
        if self.bias is not None:
            out = out + self.bias

        return out


class ResidualGATv2Layer(nn.Module):
    """
    GATv2 layer com skip connection e normalização.

    Previne over-smoothing e melhora fluxo de gradientes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.gat = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            num_heads=num_heads,
            concat=True,
            dropout=dropout
        )

        if use_layer_norm:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]

        Returns:
            out: [num_nodes, hidden_dim]
        """
        # GATv2
        out = self.gat(x, edge_index)

        # Dropout
        out = self.dropout(out)

        # Residual connection
        out = out + x

        # Normalization
        out = self.norm(out)

        return out
