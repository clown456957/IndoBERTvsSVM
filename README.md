# IndoBERT vs SVM: A Comparative Study on Sentiment Classification

![IndoBERT vs SVM](https://img.shields.io/badge/Release-Download%20Now-blue)

Welcome to the **IndoBERT vs SVM** repository! This project focuses on sentiment classification of Indonesian Twitter data using the hashtag **#KaburAjaDulu**. Here, we explore the performance of a fine-tuned **IndoBERT** model compared to traditional machine learning models, particularly **Support Vector Machines (SVM)** using IndoBERT embeddings. 

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Performance Comparison](#performance-comparison)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This repository contains the final project (skripsi) for sentiment classification. We focus on the Indonesian language, utilizing Twitter data to assess public sentiment through the lens of social media. The project primarily compares the effectiveness of a fine-tuned IndoBERT model against traditional machine learning approaches like SVM.

## Technologies Used

This project leverages several powerful tools and libraries:

- **BERT**: A state-of-the-art model for NLP tasks.
- **Hugging Face Transformers**: An extensive library for NLP tasks that simplifies the use of transformer models.
- **Python**: The programming language used for implementation.
- **Scikit-learn**: A library for machine learning that includes SVM.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/clown456957/IndoBERTvsSVM.git
   cd IndoBERTvsSVM
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   You can download the dataset from the [Releases section](https://github.com/clown456957/IndoBERTvsSVM/releases). 

## Usage

After setting up the environment, you can run the sentiment classification scripts. The main scripts include:

- **train.py**: For training the IndoBERT model.
- **svm_train.py**: For training the SVM model using IndoBERT embeddings.
- **evaluate.py**: For evaluating the models on a test dataset.

### Example Commands

To train the IndoBERT model:
```bash
python train.py --data_path path/to/dataset.csv
```

To train the SVM model:
```bash
python svm_train.py --data_path path/to/dataset.csv
```

To evaluate the models:
```bash
python evaluate.py --model_path path/to/model --data_path path/to/test_dataset.csv
```

## Performance Comparison

In this section, we will discuss the performance metrics used to compare the models. Key metrics include:

- **Accuracy**: The percentage of correct predictions.
- **Precision**: The ratio of true positives to the sum of true positives and false positives.
- **Recall**: The ratio of true positives to the sum of true positives and false negatives.
- **F1 Score**: The harmonic mean of precision and recall.

### Metrics Visualization

You can visualize the performance metrics using libraries like Matplotlib and Seaborn. 

## Results

The results section provides a comprehensive overview of how the models performed. 

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| IndoBERT      | 92%      | 91%       | 93%    | 92%      |
| SVM           | 85%      | 84%       | 86%    | 85%      |

The IndoBERT model outperformed the SVM model in all metrics, showcasing the advantages of using transformer-based models for sentiment analysis in the Indonesian language.

## Contributing

We welcome contributions! If you want to improve this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push to your branch and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please reach out to the repository maintainer:

- **Name**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [Your GitHub Profile](https://github.com/yourprofile)

For updates and releases, please check the [Releases section](https://github.com/clown456957/IndoBERTvsSVM/releases). 

Thank you for your interest in the IndoBERT vs SVM project!