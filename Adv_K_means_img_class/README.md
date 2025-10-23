# Adversarial Resilience of K-Means and ResNet-18 on MNIST

This project investigates the robustness of two different classification models—a K-Means clustering model used as a classifier and a ResNet-18-based neural network—against adversarial examples generated using the **Iterative Fast Gradient Sign Method (iFGSM)** attack. The analysis uses the **MNIST** dataset of handwritten digits.

---

## Project Structure and Workflow

The project is executed via three main Jupyter notebooks:

1.  **`ps_script.ipynb` (Poisoning/Surrogate Model):**
    * Trains a **target K-Means classifier** on the MNIST training data.
    * Trains a **surrogate ResNet-18 model** on the MNIST training data.
    * Generates **adversarial examples** by attacking the surrogate ResNet-18 model using the **iFGSM** method.
    * The generated adversarial data is then tested against the K-Means classifier to assess the transferability of the attack.
    * Saves the generated adversarial examples and their true labels to `adversarial_data.mat`.

2.  **`adv_km_script.ipynb` (K-Means Adversarial Analysis):**
    * Compares the performance of a **Normally Trained K-Means** model (trained only on original MNIST data) and an **Adversarially Trained K-Means** model (trained on a mix of original and adversarial data).
    * Analyzes the clustering of original and adversarial examples by the **adversarially trained K-Means** model.

3.  **`adv_nn_script.ipynb` (Neural Network Adversarial Analysis):**
    * Compares the performance of a **Normally Trained ResNet-18** model and an **Adversarially Trained ResNet-18** model using the adversarial data generated in `ps_script.ipynb`.

---

## Model Performance Summary

The analysis highlights the significant vulnerability of the normally trained models and the enhanced robustness of the adversarially trained ResNet-18.

### 1. K-Means Classifier

The K-Means model is highly susceptible to the transferred iFGSM attack.

| Training Method | Test Set | Original Accuracy | Adversarial Accuracy | Overall Accuracy |
| :-------------- | :------- | :---------------- | :------------------- | :--------------- |
| **Normal** | Original | 92.37%             | 0.00%                | 91.07%           |
| **Adversarial** | Combined | 91.94%             | 98.00%               | 91.77%           |

* **Normal Training:** The model, trained only on original data, achieves a respectable 92.37% on the original test set but completely fails on the adversarial examples (**0.00% accuracy**).
* **Adversarial Training:** Incorporating the adversarial examples into the training data dramatically improves performance on the adversarial set (**98.00% accuracy**). This improvement comes with a slight, almost negligible, drop in original accuracy (from 92.37% to 91.94%).

---

### 2. Neural Network (ResNet-18)

The ResNet-18 model shows a similar, but more pronounced, increase in robustness after adversarial training.

| Training Method | Test Set | Original Accuracy | Adversarial Accuracy | Overall Accuracy |
| :-------------- | :------- | :---------------- | :------------------- | :--------------- |
| **Normal** | Original | 99.01%             | 0.00%                | 87.73%           |
| **Adversarial** | Combined | 99.17%             | 100.00%              | 99.26%           |

* **Normal Training:** Achieves a high **99.01% accuracy** on original data, but the attack is **100% successful** (0.00% accuracy on adversarial data).
* **Adversarial Training:** The adversarially trained model achieves perfect classification on the adversarial set (**100.00% accuracy**) and maintains a high accuracy on the original set (**99.17% accuracy**).

---

## K-Means Cluster Analysis

An analysis of the clusters formed by the **adversarially trained K-Means** model provides insight into how the model groups the original and adversarial data.

**Parameters:**
* `CLUSTERS = 856`
* Data used: Original MNIST + iFGSM Adversarial Examples

| Cluster Type | Count | Observation |
| :--- | :--- | :--- |
| **Pure Original** | 855 | Clusters containing **only** original MNIST examples. |
| **Pure Adversarial** | 0 | Clusters containing **only** adversarial examples. |
| **Mixed** | 1 | Cluster(s) containing both original and adversarial examples. |

**Details of the Single Mixed Cluster (Cluster 60):**

| Metric | Value |
| :--- | :--- |
| Total in Cluster | 1005 |
| Adversarial in Cluster | 1000 |
| Original in Cluster | 5 |
| Percentage Adversarial | 99.50% |
| Percentage Original | 0.50% |

### Key Takeaways from Cluster Analysis

* **Limited Separation:** K-Means clustering, despite having 856 clusters, did not effectively separate the adversarial examples into their own "pure" clusters.
* **Adversarial Concentration:** Almost all ($1000/1000$) adversarial examples were grouped into a **single, highly concentrated mixed cluster (Cluster 60)**, along with a minimal number of original examples (5 examples).
* **Majority Separation:** The vast majority of the original data was well-separated into 855 "pure original" clusters.
* **Implication for Defense:** The existence of this single, dominant mixed cluster (Cluster 60) suggests that the adversarial examples—while not entirely isolated—form a very dense, distinct region in the feature space, separate from the main mass of original data. This concentration could potentially be leveraged for **adversarial detection** (e.g., flagging any sample assigned to this cluster) or for targeted **adversarial training** focusing on data points near this boundary.