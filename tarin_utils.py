from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def showImageAndTarget(train_dataset):
    # Access the dataset
    images, targets = train_dataset[0]  # Get the first image and its annotations

    # Print the image and target information
    print("Image shape:", images[0].shape)
    print("Target:", targets)
    image = images[0]
    # Convert the image tensor to a numpy array
    image_numpy = image.numpy()

    # Ensure the image has the correct shape
    if image.ndim == 3:
        image_numpy = image_numpy.transpose(1, 2, 0)  # Convert from CxHxW to HxWxC

    # Show the image
    plt.imshow(image_numpy)
    plt.show()
