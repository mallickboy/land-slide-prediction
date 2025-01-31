<h1 align="center">Landslide Prediction Using CNN (Achieved 89% F1-Score) </h1>

This repository showcases a deep learning-based solution for landslide prediction using Convolutional Neural Networks (CNN). By analyzing satellite imagery and incorporating various environmental features, this model predicts the likelihood of landslides, enabling more effective disaster management and early warning systems.

## Dataset Overview

The dataset consists of **3,799 labeled images** with associated masks, which are used for training and evaluation. Each image represents a satellite view of a region, and its corresponding mask indicates areas at risk of landslides. The dataset includes the following:

- **Image Shape**: (128, 128, 14) — The images consist of 14 channels, capturing multiple environmental variables.
- **Mask Shape**: (128, 128, 1) — Each mask is a binary image with 1 channel, representing areas with potential landslide risks.

### Used channels in the Image:
1. **Channel 1**: Blue
2. **Channel 2**: Green
3. **Channel 3**: Red
4. **Channel 7**: Near Infrared (NIR)
5. **Channel 13**: Elevation
6. **Channel 14**: Slope

### Feature Engineering
We calculated the **Normalized Vegetation Index (NDVI)** using the formula:

\[ \text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} \]

This helps to assess the vegetation health of the region, an important factor for landslide prediction. Additionally, we normalized the slope and elevation data to improve model performance.

### Training Data Features
For model training, we utilized the following 6-channel dataset:

1. **Normalized Red**
2. **Normalized Green**
3. **Normalized Blue**
4. **NDVI**
5. **Normalized Slope**
6. **Normalized Elevation**

The shape of the training data is: (128, 128, 6).

## Model Architecture and Training

We employed Convolutional Neural Networks (CNNs) for this task, which are known for their ability to extract spatial hierarchies of features from image data. The model was trained using the selected channels, and extensive experimentation with hyperparameters and architecture optimization was performed.

Here is an image depicting the model training process:

![Model Training](https://github.com/user-attachments/assets/7cabfe06-89a3-4e59-9a7b-5657659823f2)

## Model Evaluation and Comparisons

### Underfitting Model

Initially, we observed underfitting, where the model performed poorly on both the training and validation sets. This is expected when the model is too simple or lacks sufficient capacity.

![Underfitting Model](https://github.com/user-attachments/assets/fd963614-5959-4464-9bdc-b757a665ea1b)

### Hyperparameter Tuning and Finetuning

Through hyperparameter optimization and fine-tuning, the model's performance improved significantly. The next image shows the results after these adjustments:

![Finetuned Model](https://github.com/user-attachments/assets/a2fe0a11-11f6-432c-9f82-00a07b931008)

### Best Model with 89% F1-Score

After extensive fine-tuning, the best-performing model achieved an **F1-Score of 89%**, indicating a strong balance between precision and recall. This model demonstrated the highest accuracy and reliability.

![Best Model](https://github.com/user-attachments/assets/8dea7e05-2a2a-4374-a948-f51dcabd732b)

## Confusion Matrix

The confusion matrix below illustrates the model's predictive performance on the test set. It shows how effectively the model distinguishes between landslide and non-landslide areas:

![Confusion Matrix](https://github.com/user-attachments/assets/3e7398c7-a59f-4dd2-ac95-30fb00976446)

## Threshold Selection

The selection of an optimal threshold for classification plays a crucial role in balancing false positives and false negatives. After evaluating different thresholds, the following thresholds were tested:

- **Low Threshold**:

  ![Low Threshold](https://github.com/user-attachments/assets/8ac731c3-f22e-420c-8421-fc6fc12535ef)

- **High Threshold**:

  ![High Threshold](https://github.com/user-attachments/assets/909e75c4-0ca4-4654-b0d7-aec78f0b5a40)

The **50% threshold** was selected for final classification, which provided an optimal balance between sensitivity and specificity.

## Final Prediction on Labeled Data (50% Threshold)

After applying the 50% threshold, the model predicted areas at risk of landslides on the labeled dataset, as shown in the following image:

![Final Prediction on Labeled Data](https://github.com/user-attachments/assets/be92700b-3447-42df-9d61-26c106a7611c)

## Final Predictions on Unlabeled Data

The trained model was then applied to unlabeled data to predict regions at risk of landslides:

- **Prediction 1**:

  ![Prediction 1](https://github.com/user-attachments/assets/9cd823ed-4444-4df3-9f02-ec76af5de949)

- **Prediction 2**:

  ![Prediction 2](https://github.com/user-attachments/assets/5f6d5cf1-55f8-41bc-9cdf-7c5f1865a5fc)

- **Prediction 3**:

  ![Prediction 3](https://github.com/user-attachments/assets/5de18ba6-77a0-4658-b5ab-0288d5e1bccb)

## Conclusion

This project demonstrates the powerful application of **Convolutional Neural Networks (CNNs)** for landslide prediction. By leveraging satellite imagery and environmental data, we built a robust model that achieved **89% F1-Score** through careful data preprocessing, hyperparameter optimization, and model fine-tuning.

### Key Highlights:
- **Hyperparameter Tuning**: Extensive experiments were conducted to fine-tune the model, resulting in improved performance.
- **Feature Engineering**: NDVI and normalization of slope and elevation significantly contributed to the model's predictive power.
- **Threshold Selection**: The optimal threshold was selected at **50%** to balance false positives and false negatives, ensuring reliable predictions.

### Technologies Used:
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**

### Future Work:
- Expand the dataset to include more diverse environmental factors, such as rainfall and soil type.
- Incorporate temporal data to improve prediction accuracy over time.
- Explore the use of **Deep Learning** architectures like **U-Net** for better segmentation-based predictions.

---

For more details on the code implementation and model training, please explore the repository and feel free to contribute or reach out with questions.
