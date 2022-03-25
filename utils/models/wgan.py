## WGAN-GP
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__( self, channels_img: int, features_critic: int, layer_name: str="Block" ) -> None:
        super(Critic, self).__init__()
        self.layer_name = layer_name
        self.critic_net = nn.Sequential(
            self._block( channels_img, features_critic, 3, 1, 1 ),
            self._block( features_critic, features_critic * 2, 3, 1, 1 ),
            self._block( features_critic * 2, features_critic * 2, 3, 2, 1 ),
            self._block( features_critic * 2, features_critic * 4, 3, 1, 1 ),
            self._block( features_critic * 4, features_critic * 4, 3, 2, 1 ),
            nn.Conv2d( features_critic*4, 1, kernel_size=3),
        )
        return
    
    def _block( self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int ) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.Conv2d( in_channels, out_channels, kernel_size, stride, padding, bias=False ),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    
    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        return self.critic_net(x)


class Generator(nn.Module):
    def __init__( self, latent_dim: int, channels_img: int, features_gen: int, layer_name: str="Block") -> None:
        super(Generator, self).__init__()
        self.layer_name = layer_name
        self.gen_net = nn.Sequential(
            self._block( latent_dim, features_gen * 4, 3, 1, 0),
            # self._block( features_gen * 4, features_gen * 4, 3, 1, 1 ),
            self._block( features_gen * 4, features_gen * 4, 3, 1, 0 ),
            # self._block( features_gen * 4, features_gen * 2, 3, 1, 1 ),
            self._block( features_gen * 4, features_gen * 2, 3, 1, 0 ),
            # self._block( features_gen * 2, features_gen * 2, 3, 1, 1 ),
            self._block( features_gen * 2, features_gen * 2, 3, 1, 0 ),
            self._block( features_gen * 2, channels_img, 3, 1, 1 ),
            nn.ReLU(),
        )
        return
    
    def _block( self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int ) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d( in_channels, out_channels, kernel_size, stride, padding, bias=False ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        return self.gen_net(x)


def initialize_weights( model: nn.Module ) -> None:
    for m in model.modules():
        if isinstance( m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d) ):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    return


def test() -> None:
    N, in_channels, H, W = 8, 1, 9, 9
    latent_dim = 100
    x = torch.randn( (N, in_channels, H, W) )
    critic = Critic(in_channels,64)
    initialize_weights(critic)
    assert critic(x).shape == (N,1,1,1) , "Discriminator test failed"
    gen = Generator(latent_dim, in_channels, 64)
    initialize_weights(gen)
    z = torch.randn( (N, latent_dim, 1, 1) )
    assert gen(z).shape == (N, in_channels, H, W) , "Generator test failed"
    print("Success")
    return


# if __name__ == "__main__":
#     test()