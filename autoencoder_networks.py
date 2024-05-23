import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Convolutional Autoencoder for encoding and decoding images.
    """
    def __init__(self, filters, no_latent_channels=2, no_downsamples=3, image_size = 65, input_channels = 1, activation=nn.ReLU(True), start_kernel=4):
        """
        Initializes the model with the specified configuration.
        
        Parameters:
        - filters (int): Number of filters in the first convolutional layer.
        - no_latent_channels (int): Number of channels in the latent space.
        - no_downsamples (int): Number of downsampling steps in the encoder.
        - activation (nn.Module): Activation function to use in the model.
        """
        super(Autoencoder, self).__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        self.filters = filters
        self.no_downsamples = no_downsamples
        self.no_latent_channels = no_latent_channels
        self.activation = activation

        kernel_size = start_kernel

        dim = self.filters

        # Encoder
        self.downs = nn.ModuleList()
        encoder_layers = []

        # Construct the encoder layers
        for i in range(self.no_downsamples+1):
            if i == 0:
                encoder_layers.extend(self._block(self.input_channels, dim, kernel_size=kernel_size, stride=1, batchnorm=False))
            else:
                encoder_layers.extend(self._block(dim, dim, kernel_size=4, stride=2))
                encoder_layers.extend(self._block(dim, dim*2, kernel_size=3, stride=1))
                dim *= 2

            if i == self.no_downsamples:
                encoder_layers.append(nn.Conv2d(dim, self.no_latent_channels, kernel_size=3, padding=1))
            else:
                encoder_layers.extend(self._block(dim, dim, kernel_size=3, stride=1))

            self.downs.append(nn.Sequential(*encoder_layers))
            encoder_layers = []
        
        # Decoder
        self.ups = nn.ModuleList()
        decoder_layers = []

        # Construct the decoder layers
        for i in range(self.no_downsamples + 1):
            if i == 0:
                decoder_layers.extend(self._block(self.no_latent_channels, dim, kernel_size=3, stride=1))
            else:
                decoder_layers.extend(self._block(dim, dim, kernel_size=3, stride=1))
            
            if i == self.no_downsamples:
                decoder_layers.append(nn.ConvTranspose2d(in_channels=dim, out_channels=self.input_channels, kernel_size=kernel_size, padding=1))
            else:
                dim //= 2
                decoder_layers.extend(self._block(dim*2, dim, kernel_size=3, stride=1))
                decoder_layers.extend(self._upblock(dim, dim, kernel_size=4, stride=2))

            self.ups.append(nn.Sequential(*decoder_layers))
            decoder_layers = []
            
        # Initialize weights
        self.apply(self.init_weights)

    def _block(self, in_channels, out_channels, kernel_size, stride, batchnorm=True):
        """
        Helper function to create a convolutional block with optional batch normalization and activation.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(self.activation)
        return layers
        
    def _upblock(self, in_channels, out_channels, kernel_size, stride):
        """
        Helper function to create a transposed convolutional block with batch normalization and activation.
        """
        return [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation,
        ]
    
    @staticmethod
    def init_weights(m):
        """
        Initialize weights of the model.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)
    
    def encoder(self, x):
        for down in self.downs:
            x = down(x)
        return x
    
    def decoder(self, x):
        for up in self.ups:
            x = up(x)
        return x
    
    def forward(self, x):        
        x = self.encoder(x)
        activations = x
        x = self.decoder(x)
            
        return x, activations
    
    
class DeepAutoencoder(nn.Module):
    """
    Deepr Convolutional Autoencoder for encoding and decoding images.
    """
    def __init__(self, filters, no_latent_channels=2, no_downsamples=3, image_size = 65, input_channels = 1, activation=nn.ReLU(True), start_kernel=4):
        """
        Initializes the model with the specified configuration.
        
        Parameters:
        - filters (int): Number of filters in the first convolutional layer.
        - no_latent_channels (int): Number of channels in the latent space.
        - no_downsamples (int): Number of downsampling steps in the encoder.
        - activation (nn.Module): Activation function to use in the model.
        """
        super(DeepAutoencoder, self).__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        self.filters = filters
        self.no_downsamples = no_downsamples
        self.no_latent_channels = no_latent_channels
        self.activation = activation

        kernel_size = start_kernel

        dim = self.filters

        # Encoder
        self.downs = nn.ModuleList()
        encoder_layers = []

        # Construct the encoder layers
        for i in range(self.no_downsamples+1):
            if i == 0:
                encoder_layers.extend(self._block(self.input_channels, dim, kernel_size=kernel_size, stride=1, batchnorm=False))
            else:
                encoder_layers.extend(self._block(dim, dim, kernel_size=4, stride=2))
                encoder_layers.extend(self._block(dim, dim*2, kernel_size=3, stride=1))
                dim *= 2
            
            encoder_layers.extend(self._block(dim, dim, kernel_size=3, stride=1))

            if i == self.no_downsamples:
                encoder_layers.append(nn.Conv2d(dim, self.no_latent_channels, kernel_size=3, padding=1))
            else:
                encoder_layers.extend(self._block(dim, dim, kernel_size=3, stride=1))

            self.downs.append(nn.Sequential(*encoder_layers))
            encoder_layers = []
        
        # Decoder
        self.ups = nn.ModuleList()
        decoder_layers = []

        # Construct the decoder layers
        for i in range(self.no_downsamples + 1):
            if i == 0:
                decoder_layers.extend(self._block(self.no_latent_channels, dim, kernel_size=3, stride=1))
            else:
                decoder_layers.extend(self._block(dim, dim, kernel_size=3, stride=1))
            
            decoder_layers.extend(self._block(dim, dim, kernel_size=3, stride=1))
            
            if i == self.no_downsamples:
                decoder_layers.append(nn.ConvTranspose2d(in_channels=dim, out_channels=self.input_channels, kernel_size=kernel_size, padding=1))
            else:
                dim //= 2
                decoder_layers.extend(self._block(dim*2, dim, kernel_size=3, stride=1))
                decoder_layers.extend(self._upblock(dim, dim, kernel_size=4, stride=2))

            self.ups.append(nn.Sequential(*decoder_layers))
            decoder_layers = []
            
        # Initialize weights
        self.apply(self.init_weights)

    def _block(self, in_channels, out_channels, kernel_size, stride, batchnorm=True):
        """
        Helper function to create a convolutional block with optional batch normalization and activation.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(self.activation)
        return layers
        
    def _upblock(self, in_channels, out_channels, kernel_size, stride):
        """
        Helper function to create a transposed convolutional block with batch normalization and activation.
        """
        return [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation,
        ]
    
    @staticmethod
    def init_weights(m):
        """
        Initialize weights of the model.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)
    
    def encoder(self, x):
        for down in self.downs:
            x = down(x)
        return x
    
    def decoder(self, x):
        for up in self.ups:
            x = up(x)
        return x
    
    def forward(self, x):        
        x = self.encoder(x)
        activations = x
        x = self.decoder(x)
            
        return x, activations
    
    