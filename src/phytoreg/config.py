"""Configuration of the single PhytoReg architecture."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class BackboneConfig:
    input_length_bp: int = 131072
    downstream_output_length_bp: int = 65536
    token_embed_dim: int = 16
    conv_init_embed_dim: int = 512
    encoder_channels: tuple[int, ...] = (585, 658, 731, 804, 878, 951, 1024)
    transformer_dim: int = 1024
    transformer_layers: int = 9
    transformer_heads: int = 16
    transformer_ffn_dim: int = 4096
    transformer_dropout: float = 0.
    attention_dropout: float = 0.
    rotary_theta: float = 10000.
    use_skip_connection: bool = True
    gradient_checkpointing: bool = True

    @property
    def input_channels(self):
        return 5

    @property
    def bottleneck_tokens(self):
        return self.input_length_bp // 128

    @property
    def downstream_output_bins(self):
        return self.downstream_output_length_bp // 32

    def validate(self):
        positive = ('input_length_bp', 'downstream_output_length_bp', 'token_embed_dim',
                    'conv_init_embed_dim', 'transformer_dim', 'transformer_layers',
                    'transformer_heads', 'transformer_ffn_dim')
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if (self.input_length_bp % 128 or self.downstream_output_length_bp % 32
                or self.downstream_output_length_bp > self.input_length_bp
                or (self.input_length_bp - self.downstream_output_length_bp) % 64):
            raise ValueError('Invalid input length or symmetric 32-bp output crop')
        if len(self.encoder_channels) != 7 or any(
                not isinstance(c, int) or c <= 0 for c in self.encoder_channels):
            raise ValueError('Seven positive convolution widths are required')
        if self.encoder_channels[-1] != self.transformer_dim:
            raise ValueError('Last convolution width must equal transformer_dim')
        if self.transformer_dim % self.transformer_heads or (self.transformer_dim // self.transformer_heads) % 2:
            raise ValueError('Attention heads must divide transformer_dim with even head width')
        if self.rotary_theta <= 0 or not all(0 <= d < 1 for d in (self.transformer_dropout, self.attention_dropout)):
            raise ValueError('Invalid rotary base or dropout probability')

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if 'encoder_channels' in value:
            value['encoder_channels'] = tuple(value['encoder_channels'])
        config = cls(**value)
        config.validate()
        return config


@dataclass(frozen=True)
class HeadConfig:
    hidden_dim: int = 512
    output_dim: int = 256
    num_tracks: int = 1356
    registry_size: int = 1572
    num_species: int = 37
    eps: float = 1e-5

    def __post_init__(self):
        for key in ('hidden_dim', 'output_dim', 'num_tracks', 'registry_size', 'num_species'):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if self.num_tracks > self.registry_size or self.eps <= 0:
            raise ValueError('Invalid registry size or LayerNorm epsilon')


@dataclass(frozen=True)
class SpatialConfig:
    width: int = 64
    context_width: int = 16
    num_conditions: int = 52

    def __post_init__(self):
        for name in ('width', 'context_width', 'num_conditions'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class PhytoRegConfig:
    backbone: BackboneConfig
    head: HeadConfig
    region_total_length: bool = True
    spatial: SpatialConfig = SpatialConfig()

    def __post_init__(self):
        self.backbone.validate()
        if self.head.hidden_dim != self.backbone.conv_init_embed_dim:
            raise ValueError('Readout hidden_dim must equal backbone output_dim')
        if not isinstance(self.region_total_length, bool):
            raise ValueError('region_total_length must be boolean')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {f.name for f in fields(cls)}:
            raise ValueError('Expected backbone, head, spatial and region_total_length configuration')
        return cls(BackboneConfig.from_dict(value['backbone']), HeadConfig(**value['head']),
                   value['region_total_length'], SpatialConfig(**value['spatial']))

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
