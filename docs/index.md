
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

LAB6 Generative Models – conditional DDPM
---
### Overview
本實驗旨在實作一個conditional Denoising Diffusion Probabilistic Model
（Conditional DDPM），能根據多標籤條件（Multi-label conditions）生成對應
的合成影像。模型輸入為一組物件條件（例如："red sphere", "cyan cube"），根
據該條件能產生包含這些物件的圖像，並透過預訓練的ResNet18-based 
Evaluator 分類器進行準確率評估。

### Demo

#### 1. 展示不同Epoch下的訓練進度

|Epoch| Epoch 1 | Epoch 200 | Final(Ep362) |
|---|---|---|---|
|Acc |0.1022 |0.8363 |0.98611 |
|Description| 正開始從gaussian 學習 denoise| 完整重建物體但是相對位置仍在學習|幾乎準確地重建物體以及位置 完成實作|
|Gen Img|<img src="https://github.com/user-attachments/assets/66ec972b-a331-4202-af5b-9191c0633884" width="350px"/> |<img src="https://github.com/user-attachments/assets/46778cd5-9a92-4ba9-a789-3d4163f28f0e" width="350px"/> |<img src="https://github.com/user-attachments/assets/a9690ed2-d54b-4029-8f84-8ba579cd21c4" width="350px"/> |

#### 2. Denoising process

![Image](https://github.com/user-attachments/assets/20f4e1da-6f51-4289-a00e-a8614c058471)

LAB7 Policy-Based Reinforcement Learning
---

### Overview

在這次的  Lab 7  實驗中實作兩種主流的基於策略的強化學習演算法：
Advantage Actor-Critic (A2C)  和  Proximal Policy Optimization (PPO)。目的是在
瞭解這些演算法的核心機制，並在  OpenAI Gym  的經典控制任務  Pendulum-v1 
以及更具挑戰性的  MuJoCo Walker2d-v4  上評估性能。 

合作歡迎 對該研究有興趣，歡迎隨時連絡我，在LAb7 Report中，我將首先詳細描述  A2C  和  PPO (包含  GAE  和  Clipped 
Objective)  的實作，包括Network 架構、數學公式以及程式碼。接著，我會展
示並分析在三個主要任務訓練結果，特別是Training Curve、Sample Efficiency
和Training Stability。其中，我會重點比較  A2C  和  PPO  在  Pendulum  環境上
的表現差異。對於  Walker2d  任務，我還會探討  PPO  的關鍵超參數，以及使
用不同PPO 優化trick 對學習性能的影響。 
通過本次實驗，我驗證了  PPO  相較於  A2C  在Training Stability 和Sample 
Efficiency 上的優勢，並成功將  PPO  演算法應用於複雜的連續控制任務。

### Demo:專案影片展示

以下是各個演算法在不同環境下的示範影片：

#### A2C Pendulum-v1
 
https://github.com/user-attachments/assets/4e289f14-b239-4ea9-9d59-1311e98fc642

#### PPO Pendulum-v1 : 可以看到PPO 比A2C 有更穩健的控制能力

https://github.com/user-attachments/assets/b26e4064-7757-4394-a84b-ca2f23ba3150

#### PPO MuJoCo Walker2d-v4 : PPO 在 Walker2d 中搭配多個優化Trick(In Report) 可以穩健控制機器人快速行走

https://github.com/user-attachments/assets/ef00fbf5-1242-4997-bceb-fdac29779944

---
## 合作與機會

歡迎任何形式的合作與交流！我目前正尋求**強化學習（Reinforcement Learning）**領域的**實習與工作機會**。

如果您對此研究專案或相關領域有興趣，或想進一步探討合作可能性，包括：

* **實習與工作機會：** 特別是機器學習、深度學習、強化學習等相關職位。
* **學術合作：** 共同研究、論文發表、資料共享等。
* **技術交流：** 經驗分享、問題討論、新技術應用。

請隨時透過以下方式與我聯繫：

* **電子郵件：** [clu98753.cs13@nycu.edu.tw](clu98753.cs13@nycu.edu.tw)
* **LinkedIn：** [linkedin.com/in/梓誠-黃-75654b352](linkedin.com/in/梓誠-黃-75654b352)
* **GitHub Issues：** 也歡迎直接在此專案中開啟 [Issue](https://github.com/alu98753/NYCU-Deep-Learning-2025/issues/new) 進行討論。

我們期待與您交流！
---
