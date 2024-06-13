const startButton = document.getElementById('start-btn');
const nextButton = document.getElementById('next-btn');
const questionContainer = document.getElementById('question-container');
const questionElement = document.getElementById('question');
const answerButtonsElement = document.getElementById('answer-buttons');

let shuffledQuestions, currentQuestionIndex;
const questions = [

    // Chapter 5---------------------------------------------
    {
        question: "In SVM model, normalization's main benefit is to avoid having attributes in greater numeric ranges dominate those in smaller numeric ranges.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "In time-series forecasting, an estimator's mean squared error measures the average absolute error between the estimated and the actual values.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "k-NN is a prediction method used not only for classification but also for regression-type prediction problems.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Logistic regression is like linear regression where both of them are used to predict a numeric target variable.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In linear regression independence of errors assumption is also known as homoscedasticity.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Multicollinearity can be triggered by having two or more perfectly correlated explanatory variables present in the model.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Time series is a sequence of data points of interest measured and represented at consecutive and regular time intervals.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Correlation is meant to represent the linear relationships between two nominal input variables.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In linear regression, the hypothesis testing reveals the existence of relationships between explanatory (i.e., input) variables.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Homoscedasticity states that the response variables must have the same variance in their error, regardless of the explanatory variables' values.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "The _____________ method’s common idea is to split the data sample into a number of randomly drawn, disjointed subsamples.",
        answers: [
            { text: 'Cross validation', correct: true },
            { text: 'Hierarchical clustering', correct: false },
            { text: 'k-nearest neighbor', correct: false },
            { text: 'Linear regression', correct: false },
            { text: 'Clustering', correct: false }
        ]
    },
    {
        question: "Which of the following relates to a pattern-recognition methodology for machine learning.",
        answers: [
            { text: 'k-NN algorithm', correct: false },
            { text: 'Cross validation', correct: false },
            { text: 'k-means clustering', correct: false },
            { text: 'Neural computing', correct: true },
            { text: 'Naïve Bayes classifier', correct: false }
        ]
    },
    {
        question: "_____________ is a type of linear least squares method for estimating the unknown parameters in a linear regression model.",
        answers: [
            { text: 'Root mean squared error', correct: false },
            { text: 'Ordinary least squares', correct: true },
            { text: 'Mean absolute error', correct: false },
            { text: 'Mean square error', correct: false },
            { text: 'R-squared', correct: false }
        ]
    },
    {
        question: "Which of the following provides an estimate of the degree of linear association between numerically represented variables.",
        answers: [
            { text: 'Naïve Bayes', correct: false },
            { text: 'Regression', correct: false },
            { text: 'ANN', correct: false },
            { text: 'Correlation', correct: true },
            { text: 'SVM', correct: false }
        ]
    },
    {
        question: "The tasks that are followed in the SVM model when performing the data preprocessing include:",
        answers: [
            { text: 'Handling noisy values', correct: false },
            { text: 'All the answers are true', correct: true },
            { text: 'Numerisizing the data', correct: false },
            { text: 'Normalizing the data', correct: false },
            { text: 'Handling missing and incomplete data', correct: false }
        ]
    },
    {
        question: "Which of the following is not among the main assumptions in linear regression?",
        answers: [
            { text: 'Simplicity', correct: true },
            { text: 'Linearity', correct: false },
            { text: 'Independence of errors', correct: false },
            { text: 'Multicollinearity', correct: false },
            { text: 'Normality', correct: false }
        ]
    },
    {
        question: "__________ is defined as the coefficient of determination in a statistical measure of regression model.",
        answers: [
            { text: 'Mean absolute error', correct: false },
            { text: 'Ordinary least squares', correct: false },
            { text: 'Mean square error', correct: false },
            { text: 'R-squared', correct: true },
            { text: 'Root mean squared error', correct: false }
        ]
    },
    {
        question: "________________ is the occurrence of high intercorrelations among two or more independent variables in a multiple regression model.",
        answers: [
            { text: 'Independence of errors', correct: false },
            { text: 'Linearity', correct: false },
            { text: 'Normality', correct: false },
            { text: 'Multicollinearity', correct: true },
            { text: 'All the answers are true', correct: false }
        ]
    },
    {
        question: "When SVM prediction model is developed, it can be integrated into decision support system by which of the following methods?",
        answers: [
            { text: 'Graphical user interface object', correct: false },
            { text: 'ERP component', correct: false },
            { text: 'Clustering algorithm', correct: false },
            { text: 'Computational object', correct: true },
            { text: 'Nearest neighbor algorithm', correct: false }
        ]
    },
    {
        question: "_________ is used to describe the relationship between a response variable on one or more explanatory variables.",
        answers: [
            { text: 'SVM', correct: false },
            { text: 'ANN', correct: false },
            { text: 'Regression', correct: true },
            { text: 'Naïve Bayes', correct: false },
            { text: 'All of the answers are true', correct: false }
        ]
    },
    {
        question: "In Bayes theorem, the posterior probability is defined as:",
        answers: [
            { text: 'Posterior = (Likelihood * Prior) / Evidence', correct: true },
            { text: 'Posterior = (Prior * Evidence) / Likelihood', correct: false },
            { text: 'Posterior = (Likelihood * Conditional) / Evidence', correct: false },
            { text: 'None of the answers are true', correct: false },
            { text: 'Posterior = (Likelihood * Evidence) / Prior', correct: false }
        ]
    },
    {
        question: "In linear regression the relationship between the variables can be represented as:",
        answers: [
            { text: 'Mathematical equation', correct: false },
            { text: 'All the answers are true', correct: true },
            { text: 'Linear coefficient', correct: false },
            { text: 'Linear representation', correct: false },
            { text: 'Additive function', correct: false }
        ]
    },
    {
        question: "Which of the following methods cannot be used for both classification and regression type prediction problems?",
        answers: [
            { text: 'Logistic regression', correct: true },
            { text: 'SVM', correct: false },
            { text: 'All the answers are true', correct: false },
            { text: 'k-NN', correct: false },
            { text: 'ANN', correct: false }
        ]
    },
    {
        question: "When a regression equation is created between one response variable and one explanatory variable, then it is known as:",
        answers: [
            { text: 'Stepwise regression', correct: false },
            { text: 'Multiple regression', correct: false },
            { text: 'Polynomial regression', correct: false },
            { text: 'Nearest neighbor algorithm', correct: false },
            { text: 'Simple regression', correct: true }
        ]
    },


    // Chapter 6---------------------------------------------
    {
        question: "Which of the following methods is used to explain the prediction of any classifier in a human-interpretable manner by learning a surrogate model locally based on the specifics of the prediction?",
        answers: [
            { text: 'Bagging', correct: false },
            { text: 'SMOTE', correct: false },
            { text: 'LIME', correct: true },
            { text: 'Random Forest', correct: false },
            { text: 'Boosting', correct: false }
        ]
    },
    {
        question: "Which of the following from the bullseye diagram interprets the predictions that are neither consistent nor accurate?",
        answers: [
            { text: 'Low bias, high variance', correct: false },
            { text: 'High bias, low variance', correct: false },
            { text: 'Low bias, low variance', correct: false },
            { text: 'Low bias, medium variance', correct: false },
            { text: 'High bias, high variance', correct: true }
        ]
    },
    {
        question: "In which of the following model ensembles method, multiple decision trees are created from resampled data and then combine the predicted values through averaging.",
        answers: [
            { text: 'Boosting', correct: false },
            { text: 'Bagging', correct: true },
            { text: 'Stacking', correct: false },
            { text: 'Bootstrapping', correct: false },
            { text: 'All the answers are true', correct: false }
        ]
    },
    {
        question: "Local interpretability provides an explanation for an individual data point of the joint distribution of independent variables.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Which of the following techniques are used to solve the imbalanced data problems?",
        answers: [
            { text: 'Data Sampling Methods, Cost Sensitive Methods, Algorithmic Methods, Ensemble Methods', correct: true },
            { text: 'Data Sampling Methods, Cost Insensitive Methods, Algorithmic Methods, Ensemble Methods', correct: false },
            { text: 'Data Sampling Methods, Cost Sensitive Methods, Simulation Methods, Ensemble Methods', correct: false },
            { text: 'Data Sampling Methods, Cost Sensitive Methods, Data effective Methods, Ensemble Methods', correct: false },
            { text: 'Data Sampling Methods, Cost Sensitive Methods, Algorithmic Methods, Resemble Methods', correct: false }
        ]
    },
    {
        question: "Which of the following are the benefits of model ensembles?",
        answers: [
            { text: 'All the answers are true', correct: true },
            { text: 'Stable', correct: false },
            { text: 'Robustness', correct: false },
            { text: 'Accuracy', correct: false },
            { text: 'Coverage', correct: false }
        ]
    },
    {
        question: "Which of the following from bullseye diagram interprets the predictions that are inconsistent but represents a reasonably well-performing prediction model.",
        answers: [
            { text: 'Low bias, high variance', correct: true },
            { text: 'Low bias, medium variance', correct: false },
            { text: 'Low bias, low variance', correct: false },
            { text: 'High bias, high variance', correct: false },
            { text: 'High bias, low variance', correct: false }
        ]
    },
    {
        question: "Creating a good prediction model requires finding a(n) Optimal balance between the errors related to bias and variance.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Which of the following method is classified under model ensemble?",
        answers: [
            { text: 'Boosting', correct: true },
            { text: 'Clustering', correct: false },
            { text: 'SVM variants', correct: false },
            { text: 'SMOTE', correct: false },
            { text: 'Bootstrapping', correct: false }
        ]
    },
    {
        question: "Which of the following techniques creates similar minority class examples by using the k-nearest neighbor algorithm to increase the number of examples in the minority class?",
        answers: [
            { text: 'SHAP', correct: false },
            { text: 'SMOTE', correct: true },
            { text: 'Boosting', correct: false },
            { text: 'LIME', correct: false },
            { text: 'Bagging', correct: false }
        ]
    },
    {
        question: "Model ensembles are known to be more robust against outliers and noise in the data, compared to individual models.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "A data set is imbalanced when the distribution of different classes in the input variables are significantly dissimilar.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Sensitivity analysis based on input value perturbation is often used in trained feed-forward neural network modeling where all of the input variables are numeric and standardized.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Model ensembles are much easier and faster to develop than individual models.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Bagging type ensembles can be used in both regression and classification type prediction problems.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Sensitivity analysis based on leave-one-out methodology can be applied to any predictive analytics method because of its model agnostic implementation methodology.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Overfitting is the notion of making the model too specific to the training data to capture not only the signal but also the noise in the data set.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Information fusion type model ensembles utilizes meta-modeling called super learners.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In prediction modeling, reducing bias also reduces the variance.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "A model with low variance is the one that captures both noise and generalized patterns in the data and therefore produces an overfit model.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    

    {
        question: "Underfitting is mainly characterized on the bias–variance trade-off continuum as low-bias/low-variance outcome.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In explainable AI, the LIME and SHAP methods are considered as global interpreters.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In ensemble modeling, boosting builds several independent simple trees for the resultant prediction model.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In predictive analytics, the term bias commonly refers to",
        answers: [
            { text: 'Constant', correct: false },
            { text: 'Error', correct: true },
            { text: 'Accuracy', correct: false },
            { text: 'Variance', correct: false },
            { text: 'Consistency', correct: false }
        ]
    },
    {
        question: "In predictive analytics, the term variance commonly refers to",
        answers: [
            { text: 'Accuracy', correct: false },
            { text: 'Variance', correct: false },
            { text: 'Consistency', correct: true },
            { text: 'Error', correct: false },
            { text: 'Constant', correct: false }
        ]
    },
    {
        question: "Which of the following is not among the data sampling methods?",
        answers: [
            { text: 'Undersampling', correct: false },
            { text: 'Clustering', correct: true },
            { text: 'SMOTE', correct: false },
            { text: 'Oversampling', correct: false },
            { text: 'Bootstrapping', correct: false }
        ]
    },


    // chapter 7------------------------------------

    {
        question: "In the first task of the text mining process, the data is structured and preprocessed to achieve hidden patterns and knowledge nuggets.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Clustering is a supervised learning process in which objects are assigned to pre-determined number of artificial groups called clusters.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Automatic summarization is a program that is used to assign documents into a predefined set of categories.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "The main aim of NLP is to move away from word counting to a real understanding and processing of natural human language.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "SCM and ERP are the first two beneficiaries of the NLP and WordNet.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In the term-document-matrix, the columns represent the documents and the rows represent the terms, and the cells represents the frequencies.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Text-to-speech is a text processing function that can read a textual content and detects and corrects the syntactic and semantic errors.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In the context of text mining, the structured data is for the humans to process while unstructured data is for computers to process and understand.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In marketing applications, text mining can be used to assess and help predict a customer’s propensity to attrite.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Polygraph is a non-intrusive deception-detection technique commonly used to assess the level of truthfulness in textual content.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Which of the following is not among the steps involved in sentiment analysis?",
        answers: [
            { text: 'Sentiment detection', correct: false },
            { text: 'Target identification', correct: false },
            { text: 'N-P polarity classification', correct: false },
            { text: 'Collection and aggregation', correct: false },
            { text: 'Latent Dirichlet allocation', correct: true }
        ]
    },
    {
        question: "Which of the following sequence of tasks represents the text mining process?",
        answers: [
            { text: 'Establish the Corpus, Preprocess the Data, Extract the Knowledge', correct: true },
            { text: 'Establish the Corpus, Extract Knowledge, Deploy the Data', correct: false },
            { text: 'Establish the Corpus, Preprocess the Data, Build the Data', correct: false },
            { text: 'Establish the Corpus, Preprocess the Data, Develop and Test the Data', correct: false },
            { text: 'Preprocess the Data, Model the Data, Establish the Corpus', correct: false }
        ]
    },
    {
        question: "In text mining process, which of the following is not a method category used for knowledge extraction?",
        answers: [
            { text: 'Classification', correct: false },
            { text: 'Trend analysis', correct: false },
            { text: 'Clustering', correct: false },
            { text: 'Regression', correct: true },
            { text: 'Association', correct: false }
        ]
    },
    {
        question: "Which of the following applications do not utilize capabilities of text mining?",
        answers: [
            { text: 'Academic literature applications', correct: false },
            { text: 'Biomedical applications', correct: false },
            { text: 'None of the answers is true', correct: true },
            { text: 'Marketing applications', correct: false },
            { text: 'Security applications', correct: false }
        ]
    },
    {
        question: "Indices is often used to characterize the relationships between the individual terms and individual documents.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Optical character recognition is the system that automatically translates images of handwritten documents into machine-editable textual documents.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "In the context of text mining, which of the following is a part of NLP that studies the internal structure of words (i.e., the patterns of word formation within a language or across languages).",
        answers: [
            { text: 'Morphology', correct: true },
            { text: 'Corpus', correct: false },
            { text: 'Terms', correct: false },
            { text: 'Concepts', correct: false },
            { text: 'Stemming', correct: false }
        ]
    },
    {
        question: "Which of the following is not among the popular tasks performed by NLP?",
        answers: [
            { text: 'Text to speech', correct: true },
            { text: 'Foreign language reading and writing', correct: false },
            { text: 'Machine translation', correct: false },
            { text: 'Speech recognition', correct: false },
            { text: 'Speech acts', correct: false }
        ]
    },
    {
        question: "Structured data is usually organized into records with simple data values that include:",
        answers: [
            { text: 'Nominal', correct: false },
            { text: 'All of the answers are true', correct: true },
            { text: 'Ordinal', correct: false },
            { text: 'Categorical', correct: false },
            { text: 'Numeric', correct: false }
        ]
    },
    {
        question: "In the context of text mining, the large and structured set of texts that commonly stored and processed electronically and prepared for the purpose of conducting knowledge discovery is referred to as Corpus.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    }
    ,
    {
        question: "Singular value decomposition help reduce the overall structure of the term-document matrix to a lower dimensional space for further pattern/knowledge discovery.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "In text mining, associations refer to direct relationships between terms or sets of concepts.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "In which of the following knowledge extraction method the task of text categorization is achieved?",
        answers: [
            { text: 'Clustering', correct: false },
            { text: 'Association', correct: false },
            { text: 'Trend analysis', correct: false },
            { text: 'Classification', correct: true },
            { text: 'Linear regression', correct: false }
        ]
    },
    {
        question: "Which of the following are the best options available to manage the TDM matrix size?",
        answers: [
            { text: 'Principal-component analysis, labor-intensive process, and normalization.', correct: false },
            { text: 'Labor-intensive process, principal-component analysis, and log transformation.', correct: false },
            { text: 'Labor-intensive process, eliminate terms, and increase the dimensionality of input matrix.', correct: false },
            { text: 'Labor-intensive process, singular value decomposition, import terms.', correct: false },
            { text: 'Labor-intensive process, singular value decomposition, eliminate terms.', correct: true }
        ]
    },
    {
        question: "Which of the following are the most commonly used normalization methods?",
        answers: [
            { text: 'Binary, Inverse, and Direct document normalization', correct: false },
            { text: 'Log, Binary, and Exponential normalization', correct: false },
            { text: 'Log, Binary and Inverse document frequencies', correct: true },
            { text: 'Digital, Inverse, and Log document frequencies', correct: false },
            { text: 'None of the answers are true', correct: false }
        ]
    },
    {
        question: "Sentiment analysis is the technique that is used to detect the direction of opinions about specific products and/or services using large textual data sources.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "In the knowledge extraction method of text mining process, ____________ refers to the natural grouping, analysis, and navigation of large text collections, such as web pages.",
        answers: [
            { text: 'Trend analysis', correct: false },
            { text: 'Regression', correct: false },
            { text: 'Clustering', correct: true },
            { text: 'Association', correct: false },
            { text: 'Classification', correct: false }
        ]
    }
    ,
    // chapter 8 
    {
        question: "Which of the following are the main problems that can be addressed by big data analytics?",
        answers: [
            { text: 'Churn identification and customer recruiting', correct: false },
            { text: 'Improved customer service', correct: false },
            { text: 'All the answers are true', correct: true },
            { text: 'Brand management', correct: false },
            { text: 'Enhanced security capabilities', correct: false }
        ]
    },
    {
        question: "The application examples of the MapReduce includes:",
        answers: [
            { text: 'Indexing, graph analysis, text analysis, machine learning', correct: true },
            { text: 'Indexing and search, text analysis, machine learning, stored procedures', correct: false },
            { text: 'None of the answers are true', correct: false },
            { text: 'Indexing and search, graph analysis, macros, machine learning', correct: false },
            { text: 'Indexing and search, RDBMS, text analysis, machine learning', correct: false }
        ]
    },
    {
        question: "Which of the following is not a product from Apache Hadoop foundation.",
        answers: [
            { text: 'Pig', correct: false },
            { text: 'Hana', correct: true },
            { text: 'MapReduce', correct: false },
            { text: 'Hive', correct: false },
            { text: 'Hbase', correct: false }
        ]
    },
    {
        question: "The data elements in a stream is often referred as",
        answers: [
            { text: 'Quartiles', correct: false },
            { text: 'Doubles', correct: false },
            { text: 'Tuplee', correct: true },
            { text: 'Vectors', correct: false },
            { text: 'Variables', correct: false }
        ]
    },
    {
        question: "____________ is the node in a Hadoop cluster that initiates and coordinates MapReduce jobs or the processing of the data.",
        answers: [
            { text: 'Execution manager', correct: false },
            { text: 'Job tracker', correct: true },
            { text: 'Node tracker', correct: false },
            { text: 'Name supervisor', correct: false },
            { text: 'Data node identifier', correct: false }
        ]
    },
    {
        question: "Which of the following skills a Data Scientist should have:",
        answers: [
            { text: 'Communication and Interpersonal skills', correct: false },
            { text: 'All the answers are true', correct: true },
            { text: 'Creativity and out of the box thinking', correct: false },
            { text: 'Programming and Scripting', correct: false },
            { text: 'Domain expertise', correct: false }
        ]
    },
    {
        question: "Which of the following is not considered as a key component of Hadoop?",
        answers: [
            { text: 'SQL', correct: true },
            { text: 'Name node', correct: false },
            { text: 'Job tracker', correct: false },
            { text: 'Secondary nodes', correct: false },
            { text: 'HDFS', correct: false }
        ]
    },
    {
        question: "The main drawback of NoSQL functions in database processing is:",
        answers: [
            { text: 'They can’t handle large amount of data', correct: false },
            { text: 'They can’t handle batch processing', correct: false },
            { text: 'They can’t be used for querying unstructured data', correct: false },
            { text: 'None of the answers is true', correct: false },
            { text: 'They have traded ACID compliance for performance and scalability', correct: true }
        ]
    },
    {
        question: "Which of the following is not among the V’s used to define Big Data?",
        answers: [
            { text: 'Velocity', correct: false },
            { text: 'Volume', correct: false },
            { text: 'Variance', correct: true },
            { text: 'Veracity', correct: false },
            { text: 'Variety', correct: false }
        ]
    },
    {
        question: "Which of the following is not true for MapReduce?",
        answers: [
            { text: 'MapReduce has two main components, which are Mapper and Reducer', correct: false },
            { text: 'MapReduce can be used in machine learning applications', correct: false },
            { text: 'MapReduce is a programming model', correct: false },
            { text: 'MapReduce code can be written in SQL', correct: true },
            { text: 'Shuffle, sorting, and combiner are the key steps in MapReduce', correct: false }
        ]
    },
    {
        question: "Hadoop is a batch-oriented computing framework, which implies it does not support real-time data processing and analysis.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Hadoop is not just about the volume, but also processing of diversity of data types.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "MapReduce is a contemporary programming language designed to be used by computer programmers.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "A stream in a stream analytics is defined as a discrete and aggregated level of data elements.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Hadoop is the replacement for a data warehouse which stores and processes large amounts of structured data.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Among the variety of factors, the key driver for big data analytics is the business needs at any level, including strategic, tactical, or operational.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "A data scientist's main objective is to organize and analyze large amounts of data, to solve complex problems, often using software specifically designed for the task.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Big data comes from a variety of sources within the organization, including marketing and sales transaction, inventory records, financial transaction, and human resources and accounting records.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Grid computing increases efficiency, lowers total cost, and enhances production by processing computational jobs in a shared, centrally managed ordinary pool of computing resources.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "The main benefit of Hadoop is that it allows enterprises to process and analyze large volumes of structured and semi-structured data on specialized hardware.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Which of the following facts are related to Hadoop?",
        answers: [
            { text: 'Only A and B are true', correct: false },
            { text: 'Hadoop is a file management system that employs several products', correct: false },
            { text: 'Hadoop empowers analytics', correct: false },
            { text: 'Hadoop is not a single product, it is an ecosystem', correct: false },
            { text: 'All of the other answers are true', correct: true }
        ]
    },
    {
        question: "Which of the following is not considered as the key to the success of Big Data analytics?",
        answers: [
            { text: 'A clear business needs', correct: false },
            { text: 'Alignment between the business and IT strategy', correct: false },
            { text: 'The right analytical tools', correct: false },
            { text: 'Personnel with advanced analytical skills', correct: false },
            { text: 'A fact-based transaction system', correct: true }
        ]
    },
    {
        question: "The term velocity in big data analytics refers to how fast the digitized data is created and processed.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "In typical data stream mining applications, the purpose is to predict the class or value of new instances in the data stream, given some knowledge about the class membership or values of previous instances in the data stream.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Hadoop distributed file system was invented before Google developed MapReduce. Hence, the early versions of MapReduce relied on HDFS.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "A stream in a stream analytics is defined as a discrete and aggregated level of data elements.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Which of the following is not true for MapReduce?",
        answers: [
            { text: 'MapReduce has two main components, which are Mapper and Reducer', correct: false },
            { text: 'MapReduce can be used in machine learning applications', correct: false },
            { text: 'MapReduce is a programming model', correct: false },
            { text: 'MapReduce code can be written in SQL', correct: true },
            { text: 'Shuffle, sorting, and combiner are the key steps in MapReduce', correct: false }
        ]
    },
    {
        question: "A stream in a stream analytics is defined as a discrete and aggregated level of data elements.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    // chapter 9 ---------------------------------------
    {
        question: "The most popular approach in neural network is __________ allows all neurons to link the output in one layer to the next layer's input.",
        answers: [
            { text: 'Complex topology design', correct: false },
            { text: 'Feedforward-multi-layered perceptron', correct: true },
            { text: 'Recurrent neural connections', correct: false },
            { text: 'None of the answers are true', correct: false },
            { text: 'LSTM', correct: false }
        ]
    },
    {
        question: "____________ refers to computing systems that use mathematical models to emulate the human cognition process to find solutions to complex problems and situations where the potential answers can be imprecise.",
        answers: [
            { text: 'RNN', correct: false },
            { text: 'ANN', correct: false },
            { text: 'Cognitive computing', correct: true },
            { text: 'Deep learning', correct: false },
            { text: 'Representation learning', correct: false }
        ]
    },
    {
        question: "Which of the following technologies is not part of artificial intelligence?",
        answers: [
            { text: 'Machine learning', correct: false },
            { text: 'Natural language processing', correct: false },
            { text: 'Fast Fourier transformation', correct: true },
            { text: 'Bayesian belief networks', correct: false },
            { text: 'Deep learning', correct: false }
        ]
    },
    {
        question: "Which of the following are considered as key attributes that cognitive computing systems must have?",
        answers: [
            { text: 'Adaptive, Interactive, Iterative, Contextual', correct: true },
            { text: 'Adaptive, Incongruity, Interactive, Iterative, Contextual', correct: false },
            { text: 'Adaptive, Distinct, Contextual, Incongruity', correct: false },
            { text: 'Adaptive, Incongruity, Iterative, Contextual', correct: false },
            { text: 'Adaptive, Interactive, Varied, Contextual', correct: false }
        ]
    },
    {
        question: "Which of the following are the key components of a neural network?",
        answers: [
            { text: 'Neurons', correct: false },
            { text: 'Connections', correct: false },
            { text: 'Layers', correct: false },
            { text: 'Weights', correct: false },
            { text: 'All the answers are true', correct: true }
        ]
    },
    {
        question: "Which of the following is the most popular neural network learning method that applies the chain rule of calculus to compute the derivatives of functions?",
        answers: [
            { text: 'Stochastic gradient descent (SGD)', correct: false },
            { text: 'Backpropagation', correct: true },
            { text: 'Network gradients', correct: false },
            { text: 'Loss function', correct: false },
            { text: 'None of the answers are true', correct: false }
        ]
    },
    {
        question: "What are the main characteristics of convolutional neural networks?",
        answers: [
            { text: 'Having two or more layers involving a convolution weight function instead of general matrix multiplication', correct: true },
            { text: 'Having multiple layers involving a general matrix multiplication instead of a convolution weight function', correct: false },
            { text: 'Having a single layer involving a convolution weight function instead of general matrix multiplication', correct: false },
            { text: 'Having two or more layers involving a general matrix multiplication instead of a convolution weight function', correct: false },
            { text: 'Having a single layer involving a general matrix multiplication instead of a convolution weight function', correct: false }
        ]
    },
    {
        question: "The main characteristic of deep learning solutions is that they use AI to understand and organize data, predict the intent of a search query, improve the relevancy of results, and automatically tune the relevancy of results over time.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Human computer interaction is a critical component of cognitive systems that allows users to interact with cognitive machines and define their needs.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Deep learning analytics is a term that refers to the computing-branded technology platforms, such as IBM Watson, that specialize in processing and analyzing large, unstructured data sets.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "Cognitive computing has the capability to simulate human thought processes to assist humans in finding solutions to complex problems.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Multilayer perceptron type deep networks are also known as feedforward networks because the flow of information that goes through them is always forwarding, and no feedback connections are allowed.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "Delta (or an error) is defined as the difference the network weights in two consecutive iterations.",
        answers: [
            { text: 'True', correct: false },
            { text: 'False', correct: true }
        ]
    },
    {
        question: "In representation learning the emphasis is on automatically discovering the features to be used for analytics purposes.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "The term long short-term memory network refers to a network that is used to remember what happened in the past for a long enough time that it can be leveraged in accomplishing the task when needed.",
        answers: [
            { text: 'True', correct: true },
            { text: 'False', correct: false }
        ]
    },
    {
        question: "The optimization of performance in a neural network is usually done by an algorithm called __________?",
        answers: [
            { text: 'Forward propagation', correct: false },
            { text: 'Back-weight propagation', correct: false },
            { text: 'Network optimization', correct: false },
            { text: 'None of the answers are true', correct: false },
            { text: 'Stochastic gradient descent', correct: true }
        ]
    },
    {
        question: "The neural networks in which feedback connections are allowed are called ____________.",
        answers: [
            { text: 'Feedback networks', correct: false },
            { text: 'Kohonen neural networks', correct: false },
            { text: 'Recurrent neural networks', correct: true },
            { text: 'Feedforward networks', correct: false },
            { text: 'Linear neural networks', correct: false }
        ]
    },
    {
        question: "What is proper summation function of a single neuron with two inputs and corresponding weights?",
        answers: [
            { text: 'Y = W1 + W2', correct: false },
            { text: 'Y = 2(X1W1 + X2W2)', correct: false },
            { text: 'Y = X1 + X2', correct: false },
            { text: 'Y = X1W1 + X2W2', correct: true },
            { text: 'Y = X1W2 + X2W1', correct: false }
        ]
    },
    {
        question: "_____________ is a collection of neurons that takes inputs from the previous layer and converts those inputs into outputs for further processing.",
        answers: [
            { text: 'An output layer', correct: false },
            { text: 'A hidden layer', correct: true },
            { text: 'Transfer function', correct: false },
            { text: 'An input layer', correct: false },
            { text: 'Connection weights', correct: false }
        ]
    },
    {
        question: "Why is cognitive search is different from traditional search?",
        answers: [
            { text: 'Works on a narrow search space', correct: false },
            { text: 'Builds general purpose search applications', correct: false },
            { text: 'Uses advanced statistical technologies', correct: false },
            { text: 'Focuses on the syntactic nature of the searched data', correct: false },
            { text: 'Can handle a variety of data types', correct: true }
        ]
    }

];


startButton.addEventListener('click', startGame);
nextButton.addEventListener('click', () => {
    currentQuestionIndex++;
    setNextQuestion();
});

function startGame() {
    startButton.classList.add('hide');
    shuffledQuestions = questions.sort(() => Math.random() - .5);
    currentQuestionIndex = 0;
    questionContainer.classList.remove('hide');
    setNextQuestion();
}

function setNextQuestion() {
    resetState();
    showQuestion(shuffledQuestions[currentQuestionIndex]);
}

function showQuestion(question) {
    questionElement.innerText = question.question;
    question.answers.forEach(answer => {
        const button = document.createElement('button');
        button.innerText = answer.text;
        button.classList.add('btn');
        button.dataset.correct = answer.correct;
        button.addEventListener('click', selectAnswer);
        answerButtonsElement.appendChild(button);
    });
}

function resetState() {
    clearStatusClass(document.body);
    nextButton.classList.add('hide');
    while (answerButtonsElement.firstChild) {
        answerButtonsElement.removeChild(answerButtonsElement.firstChild);
    }
}

function selectAnswer(e) {
    const selectedButton = e.target;
    const correct = selectedButton.dataset.correct === 'true';
    setStatusClass(document.body, correct);
    Array.from(answerButtonsElement.children).forEach(button => {
        const isCorrect = button.dataset.correct === 'true';
        setStatusClass(button, isCorrect);
        button.disabled = true;
    });
    if (shuffledQuestions.length > currentQuestionIndex + 1) {
        nextButton.classList.remove('hide');
    } else {
        startButton.innerText = 'Restart';
        startButton.classList.remove('hide');
    }
}

function setStatusClass(element, correct) {
    clearStatusClass(element);
    if (correct) {
        element.classList.add('correct');
    } else {
        element.classList.add('wrong');
    }
}

function clearStatusClass(element) {
    element.classList.remove('correct');
    element.classList.remove('wrong');
}

// Debugging
console.log("Script loaded");
