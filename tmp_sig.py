import inspect
from src.models.vae import VAE, VAEConfig
from src.models.mdn_rnn import MDNRNN, MDNRNNConfig
for o in (VAEConfig, VAE, MDNRNNConfig, MDNRNN):
    print(f"\n{o.__name__}{inspect.signature(o.__init__)}")
print("\nVAEConfig from_dict:", hasattr(VAEConfig, "from_dict"),
      "| MDNRNNConfig from_dict:", hasattr(MDNRNNConfig, "from_dict"))
