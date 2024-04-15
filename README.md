# WomenWhoHateMen

All the experiments are performed over the subreddits "Feminism", "GenderCritical", "Incels" and "Mensrights". 
Before launching the experiments, please set a ```venv``` by installing the required packages (```pip3 install -r requirements.txt```).

In the following, a brief description of each file:

- ```python3 db_main_statistics.py``` produces and saves the statistics for each of the subreddit after the pre-processing step, i.e., #Authors, #Commenters, #Posts, #Comments, #Users, #Items.
- ```python3 db_main_toxicity.py``` generates and saves the toxicity scores associated to each subreddit content.
- ```python3 db_main_users_emotions.py``` and ```python3 db_main_posts_emotions.py``` compute and save the emotions associated to users and content, respectively, by adopting a fine-tuned version of Roberta and the Empath tool.

- ```python3 plot_emotions_graph.py``` generates and saves the plots regarding the users emotions and the graph structures.
- ```python3 plot_toxicity.py``` generates and saves the plot concerning the toxicity distribution.
- ```python3 plot_word2vec.py``` generates and saves the language-related figures.

- ```utils.py``` is a simple utility file.
