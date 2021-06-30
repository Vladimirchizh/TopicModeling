# Text data anonymization of social network users based on vector representations of their interests
Main target of the research is to provide text generation based on the topics representations to ensure high-quality anonymization of personal texts while maintaining the distribution of interests of the initial data.
The objectives of the study are 
- to try out different methods of topic modeling and labeling of the dataset and look through the quality of their results; 
- segmentation of the users according to their topics’ representation and demographic data; creating of new representations and imbedding it into the text generation pipeline.; 
- generation of fully anonymized dataset according to the obtained representations; testing and evaluation of the created pipeline 



<img src="pictures/pipeline.png" width="100%"/> (*Suggested approach for text anonymization algorithm*)

To achieve the purpose of the research (to create a generative model out of word embedding which will take into the account the main interests of social media users and will match the requirements of differential privacy) there was created a pipeline of the research. 


# TopicModeling

First it was needed to provide a good quality topic modeling.
Latest version in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b-gI2tNXYsKF88mNQ3ZdQLBZILjOeJw1)

The result of using ARTM algorithm can be presented as theta matrix (documents|topics distribution).
After labeling the obtained topics there appeared the possibility to assign the interests to the users according to their list of groups in social network.


<img src="pictures/docs_to_users.png" width="100%"/>



(The mean results for demographic groups.)

# New
Up-to-date version in Google Colab text generation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E9r72m0Fz3GloFDNEOKO4IZRcOvi-9-3?usp=sharing)
      