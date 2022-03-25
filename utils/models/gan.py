## GAN MODEL
import torch
import torch.nn as nn


class Discriminator( nn.Module ):
    def __init__( self, img_dim: int, layer_name: str="Block" ) -> None:
        super(Discriminator, self).__init__()
        self.layer_name = layer_name
        self.disc_net = nn.Sequential(
            nn.Linear( img_dim, 128 ),
            nn.LeakyReLU(0.1),
            nn.Linear( 128, 1 ),
            nn.Sigmoid(),
        )
        return
    
    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        return self.disc_net(x)


class Generator( nn.Module ):
    def __init__( self, latent_dim: int, img_dim: int, layer_name: str="Block" ):
        super(Generator, self).__init__()
        self.layer_name=layer_name
        self.gen_net = nn.Sequential(
            nn.Linear( latent_dim, 256 ),
            nn.LeakyReLU(0.1),
            nn.Linear( 256, img_dim ), # 28x28x1
            nn.ReLU(),
        )
    
    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        return self.gen_net(x)


# if __name__ == "__main__":
#     gen_model = Generator
#     disc_model = Discriminator