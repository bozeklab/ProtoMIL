import os
import pathlib
import glob
import types
import torch
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import transforms
from concurrent.futures import ThreadPoolExecutor

# MODEL_PATH = 'data/tenpercent_resnet18.ckpt'


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model


model = models.resnet18(pretrained=True)


def resnet_forward_impl(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x


model._forward_impl = types.MethodType(resnet_forward_impl, model)
model = model.cuda()
model.eval()

normalize_to_tensor_transform = transforms.Compose([
    transforms.ToTensor(),
])


class ImageDataset(Dataset):
    def __init__(self, dataset_path):
        self.files = list(str(p) for p in pathlib.Path(dataset_path).rglob('*[0-9].jpg'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = pil_loader(self.files[item])
        img = normalize_to_tensor_transform(img)
        return img


def get_dir_list(path):
    dirs = glob.glob(path, recursive=True)
    dirs.sort()
    return dirs


def process_dataset(dataset_path):
    out_file = os.path.join(dataset_path, 'embeddings.pth')
    if not os.path.isdir(dataset_path):
        print('no data for', dataset_path)
        return
    # if os.path.exists(out_file):
    #     print('skipping already processed', dataset_path)
    #     return
    print('processing', dataset_path)

    loader = torch.utils.data.DataLoader(
        ImageDataset(dataset_path),
        batch_size=100,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    embeds = []
    with torch.no_grad():
        for idx, photos in enumerate(loader):
            photos = photos.cuda()
            out = model(photos)
            out = out.cpu().detach()
            embeds.append(out)

    embeds = torch.cat(embeds, dim=0)
    torch.save(embeds, out_file)


if __name__ == "__main__":
    dataset_paths = get_dir_list('data/mito_all_patches_512/**/*.tif/')
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(process_dataset, dataset_paths)
