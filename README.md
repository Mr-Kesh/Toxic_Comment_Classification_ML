# Toxic Comment Detection Pipeline - A Python Machine Learning Project

This project applies supervised machine learning to detect harmful online comments using the Jigsaw Toxic Comment dataset from Kaggle. The goal is to classify comments as safe or toxic based on several label types, such as toxic, obscene, insult, threat, and identity hate. I treated this like a real world problem where a moderator or a website needs help sorting through a very large number of comments and wants a model that can highlight the risky ones first.

<br>

## Project Overview

- **Dataset:** about 160,000 comments with six label columns (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`)
- **Target label:** I combine all six labels into one binary target, where 1 means the comment has at least one harmful label and 0 means it is clean on all six
- **Class balance:** roughly 90 percent of comments are not toxic and about 10 percent are toxic, so the classes are very imbalanced
- **Text cleaning:** lowercase text, basic leetspeak replacement (for example `$` to `s`), remove punctuation, fix repeated letters, and trim spaces
- **Feature extraction:** character level TF IDF (n grams of length 3 to 5) to catch toxic words even when people change spellings
- **Train or test split:** stratified 80 or 20 split so both sets keep the same ratio of toxic and not toxic comments

In other words, each comment starts as raw text that might include capital letters, symbols, and creative spelling. I first clean that text so it is consistent. Then I turn it into numerical features that models can work with. Finally, I train three different models and compare how well they pick up toxic behavior across this big and imbalanced dataset.

<br>

## PCA Feature Visualization

I applied PCA on a random sample of the TF IDF features to reduce them to two dimensions so I could see how the comments are spread out in a simple plot.

<br>

![PCA of TF IDF Features](Images/PCA%20of%20TF-IDF%20Features.png)

<br>

In the PCA plot, toxic and not toxic comments overlap a lot with only a small amount of separation. Each point represents a single comment that has been compressed from tens of thousands of TF IDF features down to just two numbers. That is why the clouds of points are dense and mixed together.

Even though the classes are mixed, there is a slight shift in where toxic comments tend to appear compared to not toxic comments. This tells me that there is some structure in the data for models to learn, but it is not as simple as drawing a straight line between the two groups. It also shows why I need more flexible models that can handle complex patterns instead of just relying on basic linear separation.

<br>

## Models and Performance

---

I trained three different models on the same TF IDF or sequence features and compared how well they detected toxic comments. All three models see the same cleaned comments and try to solve the same task, but they use very different strategies to make their predictions.

| Model                              | Input features                  | Test accuracy (internal split) |
| ---------------------------------- | ------------------------------- | ------------------------------ |
| k Nearest Neighbors (kNN)          | TF IDF (20,000 sample limit)    | about 94 percent               |
| Multi Layer Perceptron (MLP)       | TF IDF (50,000 sample limit)    | about 96 percent               |
| Convolutional Neural Network (CNN) | token sequences with embeddings | about 96 percent               |

<br>

### k Nearest Neighbors (kNN)

For kNN I limited the training and test data to 20,000 examples each so it would run in a reasonable amount of time.

![Confusion Matrix kNN](<Images/Confusion%20Matrix%20(kNN).png>)

The kNN model classifies a new comment by looking at its closest neighbors in TF IDF space and taking a vote. The idea is that if a new comment looks similar to a group of known toxic comments, it should probably be considered toxic as well.

The model reached about 0.94 accuracy on my test split, but the confusion matrix showed that it still missed many toxic comments. It did well on not toxic comments but had a large number of false negatives, which means it often called toxic comments safe. This happens because the dataset is very imbalanced and because toxic comments can appear in many different forms, so a toxic comment might not have many very close neighbors for the model to copy.

<br>

### Multi Layer Perceptron (MLP)

The MLP is a small neural network that I trained on up to 50,000 TF IDF examples with early stopping.

![Confusion Matrix MLP](<Images/Confusion%20Matrix%20(MLP).png>)

The MLP works by passing the TF IDF features through layers of connected nodes and learning weights on those connections during training. Instead of only copying the behavior of nearby examples, it learns general rules about which character patterns and combinations tend to show up in toxic comments versus safe comments.

The MLP reached about 0.96 accuracy on the test set and clearly improved on the toxic class compared to kNN. The confusion matrix showed more true positives and fewer false negatives, which means it caught more harmful comments while keeping false alarms relatively low. It still makes mistakes, but it does a better job of balancing the cost of missing toxic comments against the cost of flagging clean comments.

<br>

### Convolutional Neural Network (CNN)

For the CNN I tokenized the cleaned text, padded the sequences to a fixed length, and trained an embedding based CNN for two epochs.

![Confusion Matrix CNN](<Images/Confusion%20Matrix%20(CNN).png>)

The CNN treats each comment as a sequence of tokens and learns an embedding for each token. Convolutional filters slide over this sequence and pick up short phrases and local patterns that show up often in toxic comments. This makes the CNN especially good at noticing patterns like threats, insults, or hateful phrases that might be missed if we only looked at character counts.

The CNN also reached about 0.96 accuracy on the test set. Because it looks at local word order instead of only TF IDF counts, it handled short toxic phrases and unusual spellings better and did a strong job on both classes. It was especially helpful on comments where the same word can appear in both a harmless and a harmful context, because the model can see which other words appear around it.

<br>

## Kaggle Test Evaluation

---

To test the full pipeline on data outside my own train or test split, I used Kaggle `test.csv` with its matching `test_labels.csv`. I joined them on the `id` column, dropped rows with a toxic label of `-1`, cleaned the text the same way as the training data, and ran all three models.

On this Kaggle test set I focused on precision, recall, and F1 score for the toxic class instead of only accuracy, because the data is very imbalanced.

- **Precision for toxic comments** tells me, out of all the comments the model said were toxic, how many were actually toxic. High precision means the model does not cry wolf too often and does not mark too many clean comments as toxic.
- **Recall for toxic comments** tells me, out of all the truly toxic comments in the data, how many the model actually caught. High recall means the model does not miss many toxic comments.
- **F1 score** combines precision and recall into a single number, so it rewards models that do well on both and punishes models that are only good at one of them.

Across these metrics the MLP and CNN both gave better balance on the toxic class than kNN, with higher recall and stronger F1 scores while keeping good precision. This matches what I saw in the internal confusion matrices and supports the idea that neural models are a better fit for this problem than a simple neighbor based approach.

<br>

## Key Learnings

---

- **Combining labels into one target helped define the problem clearly.** Instead of only looking at the toxic column, I combined all six harmful label columns into one overall target. This fits better with how a real moderation system might work, since a comment that is obscene or hateful is still harmful even if the toxic column by itself is 0.
- **TF IDF with character n grams is powerful for messy text.** Using character n grams instead of full words helped the models catch toxic patterns even when users tried to hide them with creative spelling. It also reduced the impact of small changes in punctuation or casing that do not really change the meaning of the comment.
- **PCA gave a helpful but limited view of the data.** The PCA plot showed that toxic and not toxic comments do not separate cleanly in two dimensions. This was useful because it set my expectations for model performance and reminded me that the real separation lives in a much higher dimensional space.
- **kNN was a good starting point but had clear limits.** kNN was easy to understand and simple to implement, but it struggled on the toxic class and produced many false negatives. This showed me that a model that only copies neighbors is not enough for a complex and imbalanced text problem like this.
- **The MLP and CNN both improved performance on toxic comments.** The MLP did a better job of learning patterns from the TF IDF features and improved recall on toxic comments without causing too many extra false positives. The CNN went a step further by looking at word order and short phrases, which made it more sensitive to the way harmful language is actually written.
- **Class imbalance changed how I evaluated the models.** Because most comments are safe, accuracy alone made all three models look strong. Looking at precision, recall, and F1 score for the toxic class gave me a much better picture of how well each model really handled the comments that matter most.

<br>

## Tech Stack

- **Language:** Python 3
- **Environment:** Jupyter Notebook
- **Libraries:** pandas, numpy, scikit learn, seaborn, matplotlib, TensorFlow or Keras
- **Models:** k Nearest Neighbors, Multi Layer Perceptron, Convolutional Neural Network

<br>

## Repository Files

- `Data/train.csv` and `Data/test.csv` or `Data/test_labels.csv` for the Jigsaw toxic comment data
- `Portfolio_Project_2_Toxic_Classification.ipynb` for the full code, exploration, and analysis
- `Portfolio_Project_2_Toxic_Classification_Report_GPT.md`

<br>

## Results

---

Overall, all three models reached high accuracy because most comments are not toxic, but the MLP and CNN gave the best balance on the toxic class itself. The CNN in particular felt the most reliable on short toxic phrases and messy spellings, while the MLP gave slightly better scores in the Kaggle classification report. Together they show that a careful text cleaning pipeline combined with modern neural models can help flag harmful online comments in a way that is both accurate and practical.