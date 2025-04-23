# Cat vs. Dog Classification: VGG16 vs. ResNet50

This repository compares two popular deep learning architectures—VGG16 and ResNet50—on the classic Cats vs. Dogs binary classification task.

## 📂 Project Structure
├── data/ │ ├── train/ # 8,000 images (4,000 cats + 4,000 dogs) │ └── test/ # 2,000 images (1,000 cats + 1,000 dogs) ├── notebooks/ │ ├── VGG16_CatsVsDogs.ipynb │ └── ResNet50_CatsVsDogs.ipynb ├── requirements.txt ├── train.py # Script to train either model ├── evaluate.py # Script to evaluate on the test set └── README.md


## 🔍 Dataset

- **Source**: [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- **Train split**: 8,000 images (4,000 cats, 4,000 dogs)  
- **Test split**: 2,000 images (1,000 cats, 1,000 dogs)

### Preprocessing
1. **Resize** images to `224×224` (to match ImageNet models).  
2. **Normalize** pixel values using ImageNet statistics:  
   ```python
   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

🏗 Architectures
1. VGG16

    Layers: 13 convolutional + 3 fully-connected

    Kernels: Stacked 3×3 convs

    Pooling: 2×2 max pooling

    Pros:

        Uniform, simple design

        Strong feature extraction

    Cons:

        138M parameters → heavy compute

        Susceptible to vanishing gradients

2. ResNet50

    Blocks: Bottleneck residual blocks (1×1 → 3×3 → 1×1 convs)

    Skip connections ease gradient flow

    BatchNorm after each conv

    Global avg-pooling before FC

    Pros:

        Only ~25.5M parameters → lighter & faster

        Mitigates vanishing gradients

    Cons:

        More complex block design
