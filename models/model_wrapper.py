from torch.nn import Module

from models.abstract import CustomModule, EmbeddingGenerator


class ModelWrapper(CustomModule, EmbeddingGenerator):
    """Wraps a pre-instanciated model into a CustomModel and EmbeddingGenerator"""

    def __init__(self, model: Module):
        self.model = model

    def __call__(self, input):
        raise NotImplementedError

    def init_optimizers(self):
        raise NotImplementedError

    def train_epoch(self, train_loader, epoch, use_cuda, log_path, **kwargs):
        raise NotImplementedError

    def validate(self, val_loader, epoch, use_cuda, log_path, **kwargs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        self.model.eval()

    def cuda(self):
        self.model.cuda()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def optim_state_dict(self):
        raise NotImplementedError

    def load_optim_state_dict(self, state_dict):
        raise NotImplementedError

    def register_hooks(self, layers):  # TODO put this and the next method in context manager
        """
        This function is not supposed to be called from outside the class.
        """
        handles = []
        embedding_dict = {}

        def get_hook(name, embedding_dict):
            def hook(model, input, output):
                embedding_dict[name] = output.detach()

            return hook

        for layer in layers:
            hook = get_hook(layer, embedding_dict)
            handles.append(getattr(self.model.features, layer).register_forward_hook(hook))

        return handles, embedding_dict

    def deregister_hooks(self, handles):
        """
        This function is not supposed to be called from outside the class.
        """
        for handle in handles:
            handle.remove()

    def get_day_embeddings(self, img, layers):
        """
        Returns deep embeddings for the passed layers inside the upper encoder.
        """
        handles, embedding_dict = self.register_hooks(layers)

        # forward pass
        self.model(img)

        self.deregister_hooks(handles)

        return embedding_dict

    def get_night_embeddings(self, img, layers):
        """
        Returns deep embeddings for the passed layers inside the upper encoder.
        """
        handles, embedding_dict = self.register_hooks(layers)

        # forward pass
        self.model(img)

        self.deregister_hooks(handles)

        return embedding_dict

