from dataclasses import dataclass, field

from safetensors.torch import load_file
from torch import Tensor, nn, stack
from transformers import PretrainedConfig, PreTrainedModel

#### These are defined to match the forward pass kwargs
MODEL_IN = 'input'
"""Input name of MulticlassProbeModel"""
MODEL_OUT = 'labels'
"""Output kwarg for MulticlassProbeModel"""


@dataclass(init=False, kw_only=True)
class MulticlassProbeConfig(PretrainedConfig):
    """
    Multiclass Linear Probe Configuration
    """
    in_features: int = None
    """Number of dimensions of the input hidden state"""
    out_features: int = None
    """Number of classes in the model's output"""
    num_submodules: int = 64
    """Number of submodules. 
    - Use 64 (one for each square) f using a state_fn based on board position, .
    - Use 1 (since it's related to a single game) if using a state_fn based on game metadata (e.g., low/high elo)
    - Use 2 if you're making predictions about each player, or each king/queen (e.g., if king is in check)
    - Use 4 if making predictions about each knight/bishop"""
    model_type: str = field(default="multiclass_linear_probe", init=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_features = kwargs.get("in_features", self.in_features)
        self.out_features = kwargs.get("out_features", self.out_features)
        self.num_submodules = kwargs.get("num_submodules", self.num_submodules)
        self.model_type = kwargs.get("model_type", self.model_type)


class MulticlassProbeModel(PreTrainedModel):
    """
    Multiclass Linear Probe.
    """
    config_class = MulticlassProbeConfig
    loss_fct = nn.CrossEntropyLoss()
    submodules: nn.ModuleList

    def __init__(self, config: MulticlassProbeConfig):
        super().__init__(config)
        self.submodules = nn.ModuleList(
            [
                nn.Linear(
                    in_features = config.in_features,
                    out_features = config.out_features,
                    bias=False, #don't use bias
                    # dtype=config.dtype,
                )
                for square in range(config.num_submodules)
            ]
        )
        # Initialize weights
        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = MulticlassProbeConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(config, *model_args, **kwargs)

        # Load the weights from the safetensors file
        safetensors_path = f"{pretrained_model_name_or_path}/model.safetensors"
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        return model

    def forward(self, input: Tensor, labels=None):
        """
        inputs.shape: [batch, model.in_features] | [model.in_features]
        labels.shape: torch.Size([batch, squares, classes])
        outputs.shape: 
        - if using batches (training): torch.Size([batch, squares, classes])
        - else: torch.Size([squares, classes])
        """
        outputs = stack([module(input) for module in self.submodules]) 

        """During Training: at this point outputs.shape is [squares, batch, out_dim] 
        because each individual submodule processes the inputs in batches. So, each
        submodule returns a tensor of shape [batch, classes]. When these are
        stacked together, they end up with a shape [squares, batch, classes].
        For instance:
            - Before fix below
                $ model(ds['train'][0]['input'].unsqueeze(0)).shape
                > torch.Size([64, 1, 3])
            - After fix below:
                $ model(ds['train'][0]['input'].unsqueeze(0)).shape
                > torch.Size([1, 64, 3])
            - In unbatched mode: 
                $ model(ds['train'][0]['input']).shape
                > torch.Size([64, 3])

        To fix this, we must check if the inputs were batched. If they were, then
        we reorder the tensor dimensions to place the batch at the front."""
        if len(input.shape) == 2: 
            # outputs = outputs.permute(1,0,2) #[batch, num_tiles, num_classes]
            outputs = outputs.permute(1,2,0) #[batch, num_classes,num_tiles]

        # If labels are provided, calculate loss
        loss = None
        if labels is not None:
              loss = self.loss_fct(outputs, labels)
            
        return (loss, outputs) if loss is not None else outputs
