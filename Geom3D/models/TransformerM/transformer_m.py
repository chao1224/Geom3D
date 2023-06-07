import torch
import torch.nn as nn

from .transformer_m_encoder import TransformerMEncoder
from .modules import LayerNorm, utils


class TransformerM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.max_nodes = args.max_nodes

        self.molecule_encoder = TransformerMEncoder(
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_nodes,
            num_segments=args.num_segment,
            use_position_embeddings=args.token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_init=args.apply_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            sandwich_ln=args.sandwich_ln,
            droppath_prob=args.droppath_prob,
            add_3d=args.add_3d,
            num_3d_bias_kernel=args.num_3d_bias_kernel,
            no_2d=args.no_2d,
            mode_prob=args.mode_prob,
        )

        self.embed_out = None
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.proj_out = None

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)


        self.embed_out = nn.Linear(
            args.encoder_embed_dim, args.num_classes, bias=False
        )

    def forward(self, batched_data, perturb=None, segment_labels=None, masked_tokens=None, **unused):

        inner_states, atom_output = self.molecule_encoder(
            batched_data,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        x = self.embed_out(x)
        x = x + self.lm_output_learned_bias

        return x, atom_output

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

# only for reference
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.token_positional_embeddings = getattr(
        args, "token_positional_embeddings", True
    )
    args.num_segment = getattr(args, "num_segment", 2)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_init = getattr(args, "apply_init", False)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)

    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)


# only for reference
def bert_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.token_positional_embeddings = getattr(
        args, "token_positional_embeddings", True
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.num_segment = getattr(args, "num_segment", 2)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_init = getattr(args, "apply_init", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    args.add_3d = getattr(args, "add_3d", False)
    args.num_3d_bias_kernel = getattr(args, "num_3d_bias_kernel", 128)
    args.no_2d = getattr(args, "no_2d", False)
    args.mode_prob = getattr(args, "mode_prob", "0.2,0.2,0.6")
    base_architecture(args)