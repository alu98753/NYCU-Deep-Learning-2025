# NYCU-Deep-Learning-2025

LAB1
---

LAB2 Binary Semantic Segmentation
---
該專案實作兩種語意分割模型進行主體與背景的分割，可在Lab2 report中找到包含UNet 與 ResNet34_UNet 的模型架構、訓練設定與最佳
化策略。

![image](https://github.com/user-attachments/assets/8dd7dd2f-8bc8-47ff-b9e9-c25fe7d77c1c)


Lab3 MaskGIT for Image Inpainting
---

本實驗旨在實作一個基於 VQGAN 與 Transformer 的 MaskGIT 模型，用於圖
像 inpainting 任務。該模型首先利用助教提供預訓練的 VQGAN 將圖像編碼
為離散的 latent token，再透過建立一個 Transformer 模型學習 token 間的上下
文關係，並利用 ieterative decoding 策略逐步還原缺失區域。Lab3 report 將詳細介紹
模型的架構、訓練策略、推論流程及實驗結果。

![image](https://github.com/user-attachments/assets/a7055abb-0603-47e9-8442-ee2261e90919)
![image](https://github.com/user-attachments/assets/019793e5-7ecc-4a03-8f1a-b6e20a6b1e27)
![image](https://github.com/user-attachments/assets/5673b989-ed15-4a64-9908-e89de8afb573)

Lab4 Conditional VAE for Video Prediction 
---
在本次作業中，我們實作了一個以 Conditional Variational Autoencoder (CVAE) 為基礎的Video Prediction模型。透過輸入一張初始影格（initial frame）與對應的 pose label 序列，模型需預測未來 629 張連續的影格，總共生成 630 張影片序列。我們採用 PSNR（Peak Signal-to-Noise Ratio） 作為主要評估準則，此任務不僅需結合 VAE 模型與時間序列資訊，還牽涉到 reparameterization trick、teacher forcing 與 KL annealing 的設定策略來幫助訓練。最終會構成一個結合時序建模與生成式學習的完整預測架構，是一個整合性極高的 generative learning 練習。

<img width="918" alt="Image" src="https://github.com/user-attachments/assets/caa3d804-cd58-4b43-9310-283586a92140" />

#### 🧠 模型預測展示

本模型接收：**1 張初始影格**（Initial Frame） + **629 張未來連續影格（label）**（此處以一張簡易圖表示），最終產生 **630 張影格組成的動態序列 GIF**。

> 💡 示意公式：`初始 Frame + 未來 Label` → `預測的結果 GIF`  
> 實際運行時，模型將依據完整 629 張影格進行時序預測。

| Initial Frame | Future Label ×629 | Predicted Result (GIF, 630 frames) |
|---------------|-------------------|-------------------------------------|
| ![](https://github.com/user-attachments/assets/87030ec3-d467-4ccb-bfa8-9873db7bd2ed) | ![](https://github.com/user-attachments/assets/37b3e6d4-ebe5-4e5a-a0a5-4cb5f88e5153) | <img src="https://github.com/user-attachments/assets/ff7696d5-b8e4-4426-af67-b047ef2a5404" width="300"/> |

LAB5 DQN
---
