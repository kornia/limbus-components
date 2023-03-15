"""Component for kornia.feature.LoFTR."""
import typing
import kornia
import torch

from limbus.core import Component, InputParams, OutputParams, InputParam, OutputParam, ComponentState


class LoFTR(Component):
    r"""LoFTR.

    Args:
        name (str): name of the component.
        pretrained (typing.Optional[str], optional): Check original documentation. Default: "outdoor".
        config (typing.Dict[str, typing.Any], optional):
            Check original documentation. Default: {
                'backbone_type': 'ResNetFPN', 'resolution': (8, 2), 'fine_window_size': 5,
                'fine_concat_coarse_feat': True,
                'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
                'coarse': {'d_model': 256, 'd_ffn': 256, 'nhead': 8,
                           'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
                           'attention': 'linear', 'temp_bug_fix': False},
                'match_coarse': {'thr': 0.2, 'border_rm': 2, 'match_type': 'dual_softmax', 'dsmax_temperature': 0.1,
                                 'skh_iters': 3, 'skh_init_bin_score': 1.0, 'skh_prefilter': True,
                                 'train_coarse_percent': 0.4, 'train_pad_num_gt_min': 200},
                'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self', 'cross'],
                         'attention': 'linear'}
                }.

    Input params:
        data (typing.Dict[str, torch.Tensor]): Check original documentation.

    Output params:
        out (typing.Dict[str, torch.Tensor]): Check original documentation.

    Original documentation
    ----------------------

    Module, which finds correspondences between two images.

    This is based on the original code from paper "LoFTR: Detector-Free Local
    Feature Matching with Transformers". See :cite:`LoFTR2021` for more details.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        config: Dict with initiliazation parameters. Do not pass it, unless you know what you are doing`.
        pretrained: Download and set pretrained weights to the model. Options: 'outdoor', 'indoor'.
                    'outdoor' is trained on the MegaDepth dataset and 'indoor'
                    on the ScanNet.

    Returns:
        Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> loftr = LoFTR('outdoor')
        >>> out = loftr(input)

    """
    class InputsTyping(InputParams):  # noqa: D106
        data: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self,
                 name: str,
                 pretrained: typing.Optional[str] = "outdoor",
                 config: typing.Dict[str, typing.Any] = {
                     'backbone_type': 'ResNetFPN',
                     'resolution': (8, 2),
                     'fine_window_size': 5,
                     'fine_concat_coarse_feat': True,
                     'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
                     'coarse': {'d_model': 256, 'd_ffn': 256, 'nhead': 8,
                                'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
                                'attention': 'linear', 'temp_bug_fix': False},
                     'match_coarse': {'thr': 0.2, 'border_rm': 2, 'match_type': 'dual_softmax',
                                      'dsmax_temperature': 0.1, 'skh_iters': 3, 'skh_init_bin_score': 1.0,
                                      'skh_prefilter': True, 'train_coarse_percent': 0.4, 'train_pad_num_gt_min': 200},
                     'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8,
                              'layer_names': ['self', 'cross'], 'attention': 'linear'}}) -> None:
        super().__init__(name)
        self._obj = kornia.feature.LoFTR(pretrained, config)
        self._callable = self._obj.forward

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the inputs of the component."""
        inputs.declare("data", typing.Dict[str, torch.Tensor])

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the outputs of the component."""
        outputs.declare("out", typing.Dict[str, torch.Tensor])

    async def forward(self) -> ComponentState:
        """Execute the forward pass of the component."""
        data = await self._inputs.data.receive()
        out = self._callable(data=data)
        await self._outputs.out.send(out)
        return ComponentState.OK
