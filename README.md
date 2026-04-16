# NET-HA-ML
Network Hardware-Aware Machine-Learning Data Classifier : Framework for Reduced Complexity Encrypted Traffic Classification at the Organization Edge

Trained on native FP8 format, it is meant to be a step forward in the practicalization of Encrypted Traffic Classification in High-Speed networks (> 1Gbps).

It utilizes a 1-D CNN spatial recognition model along with a FT-Transformer for fine-grained temporal extraction.

For this, the ISCX-VPN2016 dataset, in both CSV and PCAP formats have been employed and evaluated, with PCAPs showing promising benefits but with higher training and inference complexity. Although modern hardware could run it without major issues, future hardware might be necessary for enterprise/large scale deployment.
