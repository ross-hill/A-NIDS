# A-NIDS
An Anomaly Based Network Intrusion Detection System (A-NIDS) that uses Unsupervised Learning.

## File Description

There are three python files in this repo:
  1. model.py
  2. data.py
  3. Autoencoder.py
  
 ## model.py
 This file contains the model, that combines an Isolation Forest, Random Forest and Autoencoder Neural Networks. The file should run from the command line, as it prompts the user to enter the locations of any files and any constants such as random seeds etc. It contains a section on the KDD data, in which the data is processed and a random forest is fitted. The other parts do not require any specific formatted data.
 
 ## data.py
This file contains the preprocessing of the data. This does not include the engineering of certain features such as `count` as this was done before so.

## Autoencoder.py
The most customised section, this file contains the PartialAutoencoder and EnsemblePartialAutoencoders classes that combines the autoencoder theory in the paper.
