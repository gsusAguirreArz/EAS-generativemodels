# INFOGAN
import torch
from torch import nn,Tensor


class Generator( nn.Module ):
    def __init__(
        self,
        latent_dim: int,
        channels_img: int,
        features_gen: int,
        num_classes: int,
        img_size: int,
        code_dim: int,
        layer_name: str="Block"
    ) -> None:
        super( Generator, self ).__init__()
        self.init_size = img_size // 4
        self.layer_name = layer_name

        self.gen_input_layer = nn.Sequential(nn.Linear(latent_dim+num_classes+code_dim, 128 * self.init_size ** 2))
        self.gen_conv_block = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            self._block(128,128,3,1,1),
            nn.Upsample(scale_factor=2),
            self._block(128,64,3,1,1),
            nn.Conv2d(64, channels_img, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        return
    

    def _block( self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int ) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding ),
            nn.BatchNorm2d(out_channels, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
    

    def forward( self, noise: Tensor, labels: Tensor, code: Tensor ) -> Tensor:
        x = torch.cat([noise, labels, code], -1)
        x = self.gen_input_layer(x)
        x = x.view( x.shape[0], 128, self.init_size, self.init_size )
        return self.gen_conv_block(x)


class Discriminator( nn.Module ):
    def __init__(
        self,
        img_size: int,
        channels_img: int,
        num_classes: int,
        code_dim: int,
        layer_name: str="Block"
    ) -> None:
        super( Discriminator, self ).__init__()
        ds_size = img_size // 2 ** 4
        self.layer_name = layer_name

        self.disc_conv_block = nn.Sequential(
            self._block(channels_img, 16, bn=False),
            self._block(16, 32),
            self._block(32, 64),
            self._block(64, 128),
        )

        self.disc_adv_output_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.disc_aux_output_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, num_classes), nn.Softmax())
        self.disc_latent_output_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))

        return
    

    def _block( self,  in_channels: int, out_channels: int, bn: bool=True ) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.Conv2d( in_channels, out_channels, 3, 2, 1 ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(out_channels, 0.8),
        )
    

    def forward( self, x: Tensor ) -> Tensor:
        x = self.disc_conv_block(x)
        x = x.view(x.shape[0], -1)
        return self.disc_adv_output_layer(x), self.disc_aux_output_layer(x), self.disc_latent_output_layer(x)


def initialize_weights( model: nn.Module ) -> None:
    for m in model.modules():
        if isinstance( m, (nn.Conv2d, nn.ConvTranspose2d, ) ):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance( m, (nn.BatchNorm2d, )):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    return