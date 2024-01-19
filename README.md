Simple Unet example using keras for image segmentation

needs 
https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



Epochs processed 50 / 50
------------------Accuracy------------------
Class 0 Got 1602823/1672800 with accuracy 95.82
Class 1 Got 1648523/1672800 with accuracy 98.55
Class 2 Got 1668667/1672800 with accuracy 99.75
Class 3 Got 1615118/1672800 with accuracy 96.55
--------------------Dice--------------------
Class 0 Dice score 90.96
Class 1 Dice score 93.17
Class 2 Dice score 98.63
Class 3 Dice score 97.09


Epochs processed 500 / 500
------------------Accuracy------------------
Class 0 Got 1613421/1672800 with accuracy 96.45
Class 1 Got 1653440/1672800 with accuracy 98.84
Class 2 Got 1669238/1672800 with accuracy 99.79
Class 3 Got 1622857/1672800 with accuracy 97.01
--------------------Dice--------------------
Class 0 Dice score 92.08
Class 1 Dice score 94.45
Class 2 Dice score 98.83
Class 3 Dice score 97.44


Performance training and validation set
------------------Accuracy------------------
Class 0 Got 5502636/5573280 with accuracy 98.73
Class 1 Got 5550683/5573280 with accuracy 99.59
Class 2 Got 5569571/5573280 with accuracy 99.93
Class 3 Got 5513671/5573280 with accuracy 98.93
--------------------Dice--------------------
Class 0 Dice score 97.25
Class 1 Dice score 98.13
Class 2 Dice score 99.64
Class 3 Dice score 99.07
Performance test set
------------------Accuracy------------------
Class 0 Got 591103/628320 with accuracy 94.08
Class 1 Got 613588/628320 with accuracy 97.66
Class 2 Got 620343/628320 with accuracy 98.73
Class 3 Got 603711/628320 with accuracy 96.08
--------------------Dice--------------------
Class 0 Dice score 89.28
Class 1 Dice score 89.62
Class 2 Dice score 94.27
Class 3 Dice score 96.05