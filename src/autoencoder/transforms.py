# To use the dataset with a DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from autoencoder.dataset import ImageDataset


def view_tiff(
    tiff_path="/home/arelius/workspace/autoencoder/data/NIH-NLM-ThickBloodSmearsU/Uninfected Patients/TF201_HT6/tiled/20170728_201358.tiff",
):
    import matplotlib.pyplot as plt
    from PIL import Image

    # Open the image file
    img = Image.open(tiff_path)
    # Display the image
    plt.imshow(img)
    plt.show()


def get_dataset(root_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageDataset(root_dir=root_dir, transform=transform)
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset


def view_batched_images(data_loader_in, data_loader_un):
    # Iterate over the data loaders and view image
    for i, data_loader in enumerate([data_loader_in, data_loader_un]):
        # Get the first batch of images and labels
        images, labels = next(iter(data_loader))

        # Get the first image from the batch
        image = images[0]

        # Convert the image tensor to a PIL Image

        # Convert tensor to PIL Image
        to_pil = transforms.ToPILImage()
        image = to_pil(image)

        # Display the image
        plt.figure(i)
        plt.imshow(image)
        plt.title(f"DataLoader {i+1}, Batch 1, Image 1")
        plt.show()
