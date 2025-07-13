CNN Model version 1.0 - no transfer learning, with data augmentation 50 Epochs with early stopping - 68 - 70% test accuracy
CNN Model version 2.0 -> transfer learning(resnet18) + data augmentation + increased layer depths - 89-90% test accuracy => (note : if you're utilizing all your cpu cores, this will turn out be a bottleneck)
CNN Model version 3.0 ->