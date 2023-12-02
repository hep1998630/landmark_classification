import torch
import torch.nn as nn



class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.block1= nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'), # >> 16,224,224
            # nn.Dropout(dropout),
            nn.BatchNorm2d(16),
            
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.MaxPool2d(2, 2), # >> 16x112x112
            
            # Block 2
            nn.Conv2d(16, 32, 3, padding='same'),  # -> 32x112x112
            # nn.Dropout(dropout),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),  # -> 32x56x56
            
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x56x56
            # nn.Dropout(dropout),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),  # -> 64x28x28
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            # Block 4
            nn.Conv2d(64, 128, 3, padding=1),  # -> 64x56x56
            # nn.Dropout(dropout),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),  # -> 64x28x28     

            # Block 5
            nn.Conv2d(128, 128, 3, padding=1),  # -> 16x14x14
            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),  # -> 256x7x7
            
            # Block 5
            nn.Conv2d(128, 64, 3, padding=1),  # -> 16x14x14
            
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),  # -> 256x7x7  
            nn.Dropout(dropout),       
        )

        self.avg_pool= nn.AdaptiveAvgPool2d((1,1))

        self.fc_layer = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),  # -> 1x256x7x7
            
            nn.Linear(64, 500),  # -> 2000
            nn.Dropout(dropout),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            # # nn.Dropout(dropout),
            # # nn.Linear(2000, 200),
            # # nn.ReLU(),
            nn.Linear(500, num_classes),
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x= self.block1(x)

        x1 = self.avg_pool(x)

        x2 = self.avg_pool(self.block2(x))

        x= x1 + x2 



        x= self.fc_layer(x)

        return x







# # define the CNN architecture
# class MyModel2(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

#         super(MyModel2, self).__init__()

#         # YOUR CODE HERE
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))



#         self.model = nn.Sequential(
#             # Block 1
#             nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding='same'), # >> 16,224,224
#             nn.BatchNorm2d(10),
            
#             nn.ReLU(),
#             # nn.Dropout(dropout),
#             nn.MaxPool2d(2, 2), # >> 16x112x112
            
#             # Block 2
#             nn.Conv2d(10, 16, 3, padding='same'),  # -> 32x112x112
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             # nn.Dropout(dropout),
#             nn.MaxPool2d(2, 2),  # -> 32x56x56
            
#             # Block 3
#             nn.Conv2d(16, 20, 3, padding=1),  # -> 64x56x56
#             nn.Dropout(dropout),
#             nn.BatchNorm2d(20),
#             nn.ReLU(),
#             # nn.Dropout(dropout),
#             nn.MaxPool2d(2, 2),  # -> 64x28x28


#             # Block 4
#             nn.Conv2d(20, 10, 3, padding=1),  # -> 64x56x56
#             nn.Dropout(dropout),
#             nn.BatchNorm2d(10),
#             nn.ReLU(),
#             # nn.Dropout(dropout),
#             nn.MaxPool2d(2, 2),  # -> 64x28x28            

#             # Block 5
#             nn.Conv2d(10, 3, 3, padding=1),  # -> 16x14x14
#             nn.BatchNorm2d(3),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.MaxPool2d(2, 2),  # -> 256x7x7            
            


#             nn.Flatten(),  # -> 1x256x7x7
            
#             # nn.Linear(3 * 7 * 7, 1500),  # -> 2000
#             # nn.Dropout(dropout),
#             # nn.BatchNorm1d(1500),
#             # nn.ReLU(),
#             # # nn.Dropout(dropout),
#             # # nn.Linear(2000, 200),
#             # # nn.ReLU(),
#             nn.Linear(3 * 14 * 7, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)


#         return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
