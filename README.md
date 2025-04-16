# NYCU-Deep-Learning-2025

### LAB1


### LAB2 Binary Semantic Segmentation

該專案實作兩種語意分割模型進行主體與背景的分割，可在Lab2 report中找到包含UNet 與 ResNet34_UNet 的模型架構、訓練設定與最佳
化策略。

![image](https://github.com/user-attachments/assets/8dd7dd2f-8bc8-47ff-b9e9-c25fe7d77c1c)


### Lab3 MaskGIT for Image Inpainting

本實驗旨在實作一個基於 VQGAN 與 Transformer 的 MaskGIT 模型，用於圖
像 inpainting 任務。該模型首先利用助教提供預訓練的 VQGAN 將圖像編碼
為離散的 latent token，再透過建立一個 Transformer 模型學習 token 間的上下
文關係，並利用 ieterative decoding 策略逐步還原缺失區域。Lab3 report 將詳細介紹
模型的架構、訓練策略、推論流程及實驗結果。

![image](https://github.com/user-attachments/assets/a7055abb-0603-47e9-8442-ee2261e90919)
![image](https://github.com/user-attachments/assets/019793e5-7ecc-4a03-8f1a-b6e20a6b1e27)
![image](https://github.com/user-attachments/assets/5673b989-ed15-4a64-9908-e89de8afb573)
