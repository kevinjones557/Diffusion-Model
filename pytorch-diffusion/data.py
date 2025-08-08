import torchvision.transforms as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
 
def get_dataset(dataset_name, split):
    assert(split in ['train', 'validation'])
    train = split == 'train'
    # no random horizontal flip if using MNIST
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((32, 32), 
                    interpolation=TF.InterpolationMode.BICUBIC, 
                    antialias=True),
            TF.RandomHorizontalFlip(),
            TF.Lambda(lambda t: (t * 2) - 1) # scale between [-1, 1] 
        ]
    )
     
    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(root="data", train=train, download=True, transform=transforms)
    elif dataset_name == "Cifar-10":    
        dataset = datasets.CIFAR10(root="data", train=train, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(root="data", train=train, download=True, transform=transforms)
 
    return dataset
 
def inverse_transform(tensors):
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0

def get_dataloader(dataset_name, shuffle=True, batch_size=32, split='train'):
    assert(split in ['train', 'validation'])
    dataset = get_dataset(dataset_name, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
