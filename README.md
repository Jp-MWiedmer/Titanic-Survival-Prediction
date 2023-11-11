# Titanic-Survival-Prediction
EN: Machine Learning project to predict the survivors of the Titanic disaster. Uses the Titanic dataset. A SVM classifier was employed, obtaining 77% of accuracy on test set from Kaggle. 

The main file has functions and pipelines for 

1. Data preparation
2. Model selection
3. Hiperparameter adjustment
4. Model saving
5. Cross-fold validation
6. Error analysis
7. Test data prediction

The error_analysis_ file contains functions which automatize error analysis and model evaluation of binary classification, such as confusion matrix plotting, ROC curve,
Precision x Recall curves and decision threshold adjustment in order to meet Precision or Recall requirements.

PT: Projeto de Machine Learning que prevê os sobreviventes do desastre do Titanic (usando a base de dados Titanic). Utilizou-se um classificador SVM, o qual obteve acurácia de 77% 
no conjunto de teste do Kaggle.

O arquivo main contém funções e pipelines para:

1.Preparação de dados
2.Seleção de modelo
3.Ajuste de hiperparâmetros
4.Salvamento de modelo
5.Validação cruzada
6.Análise de erros
7.Previsão de dados de teste

O arquivo error_analysis_ contém funções que automatizam a análise de erros e a avaliação do modelo de classificação binária, como a plotagem da matriz de confusão, a curva ROC, 
as curvas Precisão x Sensibilidade e o ajuste do limiar de decisão para atender aos requisitos de Precisão ou Sensibilidade.

