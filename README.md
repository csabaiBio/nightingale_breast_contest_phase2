# csabAIbio 1st place solution for Nightingale High Risk Breast Cancer Prediction Contest Phase 2

## Brief summary of our approach:

For addressing the problem of stage prediction in breast cancer biopsies, our approach is a hierarchical approach, which consists of two stage low level ViT feature extractors and a third, high level fitting over biopsy bags (collection of slides). This builds heavily upon the HIPT method developed at mahmoodlab (https://github.com/mahmoodlab/HIPT). We have done multiple experiments (ResNet50 and Vision Transformer (ViT) small feature extractors) on multiple levels (0 and 1). Training were performed on biopsy bags as a multi-class classification problem. 


Furher external resources used:

- CLAM (https://github.com/mahmoodlab/CLAM)

- DINO (https://github.com/facebookresearch/dino)
