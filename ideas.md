### Project Ideas

1. Training with different loss functions like dice loss, log cosh dice loss...
2. Training for multiple epochs with less data (see if it overfits)
3. DINO regularization of clip embeddings 
4. We could try training the grounding module with fp16 and see if performance degrades. If not we halved the dataset size
5. We only need the UNet features for a given timestep. Instead of saving the features dict we could save the UNet input at that timestep and do a single forward pass to get the features out. This would significantly reduce the dataset size
6. Look into [this paper](https://github.com/fudan-zvg/GSS)
7. Maybe we could have a cheap SAM model if we add one step of noise to an image and do a forward pass of the UNet