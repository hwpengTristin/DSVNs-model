# DSVNs_model
We present an effective Disentangled Spectrum Variations Networks (DSVNs) for NIR-VIS HFR. Two key strategies are introduced to the DSVNs for disentangling spectrum variations between two domains: Spectrum-adversarial Discriminative Feature Learning (SaDFL) and Step-wise Spectrum Orthogonal Decomposition (SSOD). The SaDFL consists of Identity-Discriminative subnetwork (IDNet) and Auxiliary Spectrum Adversarial subnetwork (ASANet). Both IDNet and ASANet can jointly enhance the domain-invariant feature representations via an adversarial learning. The SSOD is built by stacking multiple modularized mirco-block DSV, and thereby enjoys the benefits of disentangling spectrum variation step by step.

# Usage
## project list

support.py

DSVNs_Architecture.py

similarity_calculation.py

README.md

## model parameters

model-DSVNs.ckpt.data-00000-of-00001

model-DSVNs.ckpt.index

model-DSVNs.meta

## Test the DSVNs network
One can test the DSVNs network by runing DSVNs_model_test.py. 

# The trained DSVNs models

## The CASIA NIR-VIS 2.0 dataset
We first pre-train the IDNet using MS_Celeb_1M dataset, and then fine-tune the DSVNs using CASIA NIR-VIS 2.0 dataset. The trained model (one of the tenfold) can be found here (Release after the paper has been accepted). The DSVNs achieves rank-1 accuracy of 99.0 ± 0.3 and VR@FAR=0.1%(%) of 98.6 ± 0.3 on CASIA NIR-VIS 2.0, respectively.

## The Oulu-CASIA NIR-VIS dataset
We first pre-train the IDNet using MS_Celeb_1M dataset, and then fine-tune the DSVNs using Oulu-CASIA NIR-VIS dataset. The DSVNs achieves rank-1 accuracy of 100% and VR@FAR=0.1%(%) of 95.5 on Oulu-CASIA NIR-VIS, respectively.

# Requirements
tensorflow 1.3.0 + 

cvxopt 1.2.0 

scipy 1.0.0 

scikit-learn 0.19.1 

# Backbone network
The inception-resnet-v1 network structure can be found here (https://github.com/davidsandberg/facenet). 

# Joint Bayesian classifier
The Joint Bayesian classifier can be found here (http://jiansun.org/papers/ECCV12_BayesianFace.pdf).

# Note
Part of our code is based on Github's open source project (https://github.com/davidsandberg/facenet).

# Reference
[1] W. P. Hu and H. F. Hu, "Disentangled Spectrum Variations Networks for NIR-VIS Face Recognition," under review, 2019

[2] F. Schroff, D. Kalenichenko and J. Philbin, "Facenet: A unified embedding for face recognition and clustering," IEEE Conf. Computer Vision and Pattern Recognition, 2015, pp. 815-823.

[3] D. Chen, X. D. Cao, L. W. Wang, F. Wen and J. Sun, "Bayesian face revisited: a joint formulation,". Springer European Conference on Computer Vision, 2012, pp. 566-579.

[4] C. Szegedy, S. Ioffe, V. Vanhoucke and A. A. Alemi, "Inception-v4, inception-resnet and the impact of residual connections on learning," arXiv preprint arXiv:1602.07261, 2016.
