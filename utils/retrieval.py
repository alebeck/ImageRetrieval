from torch import cuda
from torch.nn.functional import triplet_margin_loss
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder
from utils.config import RetrievalConfig


def evaluate(encoder_anchor: Autoencoder, encoder_opposite: Autoencoder, config: RetrievalConfig):

    triplet_loader = DataLoader(
        config.dataset(**config.dataset_args),
        batch_size=config.batch_size,
    )

    # check cuda availability
    if cuda.is_available():
        print('Using GPU...')
        encoder_anchor.cuda()
        encoder_opposite.cuda()

    encoder_anchor.eval()
    encoder_opposite.eval()

    loss_sum = 0

    for anchor_img, positive_img, negative_img in triplet_loader:

        anchor = encoder_anchor.encode(anchor_img)
        positive = encoder_opposite.encode(positive_img)
        negative = encoder_opposite.encode(negative_img)

        loss_sum += triplet_margin_loss(anchor, positive, negative)

    loss_mean = loss_sum / len(triplet_loader)
    return loss_mean
