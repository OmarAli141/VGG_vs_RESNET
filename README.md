# Cat vs. Dog Classification: VGG16 vs. ResNet50

This repository compares two popular deep learning architectures—VGG16 and ResNet50—on the classic Cats vs. Dogs binary classification task.

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



📊 Performance Comparison
Model	Training       Loss (Final)	    Test Accuracy	  Parameters	   Training Time (50 Epochs)
VGG16	                 0.2226 	         88.48%        	138M	           ~2 hours (GPU)
ResNet50	              0.0277	            73.21%	      25.5M	           ~1.5 hours (GPU)


Key Observations:

    VGG16 outperformed ResNet50 (~88% vs. ~73% accuracy).

        Possible reasons:

            VGG's deeper feature extraction works well for this smaller dataset.

            ResNet may need more data to leverage its depth effectively.

    ResNet trained faster (fewer params + skip connections help gradient flow).

    VGG had higher final loss but better generalization (possibly due to dropout layers).
