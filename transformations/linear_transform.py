import torch
import torch as th
import abc


class SingleLayerMLP(th.nn.Module):
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        # Apply L2 normalization to each embedding
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


class LinearTransform(abc.ABC):
    def __init__(self, args):
        """
        Initialize the encoder. Subclasses can implement specific initialization as needed.
        """
        
        self.transform_model = SingleLayerMLP(512, 512, normalize=True)
        dict = torch.load(args.transform_model_path)
        self.transform_model.load_state_dict(dict['model_state_dict'])
        self.transform_model.eval().cuda()


    def apply_transform(self, embedding):
        """
        Encodes a text input into a representation.
        :param text: Text data to be encoded.
        :return: Encoded representation of the text.
        """
        return self.transform_model(embedding)